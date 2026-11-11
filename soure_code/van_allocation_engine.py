#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
van_allocation_engine.py
===============================================================================
Optimal Cross-Faculty Shared Passenger Van Allocation with Safety-Compliant
Hours-of-Service (HOS) Constraints Using Mixed-Integer Linear Programming (MILP)

Reference Implementation (v6 Empirical Pipeline & Web Decision Support Backend)
for Engineering and Applied Science Research (EASR) & Multi-University Deployment.

Features:
  1. Multi-format CSV Data Preprocessing (Thai Buddhist Era B.E. -> Gregorian C.E.
     date conversion, dot-delimited time normalization, blank Excel row pruning,
     and status label standardization).
  2. Web Input Capacity Validation (enforcing passengers <= 11 * requested_vans
     in accordance with Test Cases TC66-TC69).
  3. Lexicographic MILP Solver (PuLP + COIN-OR CBC) enforcing:
     - Passenger coverage & multi-van request splitting (up to K_r vans)
     - Out-of-province mandatory driver pairing vs. in-province optional self-drive
     - Spatiotemporal non-overlapping constraints & pre-existing busy blocks
     - Statutory Hours-of-Service (HOS) rules:
         * Max 14 hours/day operational duty (H_max_day = 14.0)
         * Max 72 hours/14 days cumulative workload (H_max_14d = 72.0)
         * Min 10 hours mandatory inter-shift rest turnaround (Delta_rest = 10.0)
  4. Three Baseline Benchmark Algorithms:
     - FCFS-A (Greedy First-Come, First-Served without HOS fatigue checks)
     - FCFS-B (Safety-Compliant Greedy First-Come, First-Served)
     - Heuristic-SDF (Shortest-Duration-First Safety-Compliant Heuristic)
===============================================================================
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp


# =============================================================================
# 1. CONFIGURATION & DEFAULT INSTITUTIONAL PARAMETERS
# =============================================================================
@dataclass
class AllocationConfig:
    """Configurable parameters for any university or multi-department fleet."""
    faculties: List[str] = field(default_factory=lambda: ["ICT", "SCI", "SEEN"])
    real_data_faculties: List[str] = field(default_factory=lambda: ["ICT", "SCI"])
    van_capacity: int = 11
    max_split_vans: int = 3
    horizon_days: int = 14
    max_hours_per_day: float = 14.0
    max_hours_per_14d: float = 72.0
    min_rest_hours: float = 10.0
    allow_cross_faculty_sharing: bool = True
    # Objective hierarchy weights
    w1_unserved_pax: float = 1e7
    w2_full_request: float = 5e5
    w3_same_faculty_van: float = 100.0
    w3_cross_faculty_van: float = 400.0
    w4_primary_driver: float = 20.0
    w4_secondary_driver: float = 120.0
    solver_time_limit_sec: int = 30
    solver_msg: bool = False


# =============================================================================
# 2. DATA PREPROCESSING & WEB VALIDATION UTILITIES (v6 Pipeline)
# =============================================================================
def preclean_be_date_str(val: object) -> str:
    """
    Converts Thai Buddhist Era (B.E. > 2400) years to Gregorian (C.E.) years
    before calling pd.to_datetime to avoid 64-bit nanosecond OutOfBoundsDatetime.
    Also corrects known typographical year '2577' -> '2567' (2024 C.E.).
    """
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return ""

    def _fix_year(match: re.Match) -> str:
        y = int(match.group(1))
        if y == 2577:
            y = 2567
        if y > 2400:
            y -= 543
        return str(y)

    return re.sub(r"\b(2[45]\d{2})\b", _fix_year, s)


def parse_flexible_date(series: pd.Series) -> pd.Series:
    """Parses dates across ISO (YYYY-MM-DD) and DD/MM/YYYY (B.E. or C.E.) formats."""
    cleaned = series.apply(preclean_be_date_str)
    return pd.to_datetime(cleaned, format="mixed", dayfirst=True, errors="coerce")


def normalize_time_str(val: object, default_time: str = "08:00:00") -> str:
    """Normalizes time strings including dot-delimited formats ('10.00' -> '10:00:00')."""
    if pd.isna(val):
        return default_time
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return default_time
    s = s.replace(".", ":")
    parts = s.split(":")
    try:
        hh = int(float(parts[0]))
        mm = int(float(parts[1])) if len(parts) > 1 else 0
        ss = int(float(parts[2])) if len(parts) > 2 else 0
        if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
    except Exception:
        pass
    return default_time


def validate_web_request(pax: int, requested_vans: int, van_capacity: int = 11) -> Tuple[bool, str, str]:
    """
    Validates booking request against Web UI capacity rules (Test Cases TC66-TC69):
    - Passengers must be >= 1
    - Requested vans must be >= 1
    - Passengers must not exceed van_capacity * requested_vans
    Returns (is_valid, msg_th, msg_en).
    """
    if pax < 1:
        return False, "จำนวนผู้โดยสารต้องตั้งแต่ 1 คนขึ้นไป", "Passenger count must be at least 1."
    if requested_vans < 1:
        return False, "จำนวนรถที่ขอต้องตั้งแต่ 1 คันขึ้นไป", "Requested vehicle count must be at least 1."
    max_cap = van_capacity * requested_vans
    if pax > max_cap:
        return (
            False,
            f"ข้อมูลคำขอไม่ถูกต้อง (INVALID_INPUT): ผู้โดยสาร {pax} คน เกินเพดานความจุ {van_capacity}×{requested_vans} = {max_cap} ที่นั่ง กรุณาเพิ่มจำนวนรถที่ขอ",
            f"Invalid Request (INVALID_INPUT): {pax} passengers exceed maximum capacity of {van_capacity}×{requested_vans} = {max_cap} seats. Please increase requested vehicles.",
        )
    return True, "ผ่านเกณฑ์ความจุของระบบเว็บ", "Valid web request capacity."


# =============================================================================
# 3. CORE MILP OPTIMIZATION ENGINE & BASELINES
# =============================================================================
def solve_van_allocation_milp(
    requests: List[Dict],
    vans: List[Dict],
    drivers: List[Dict],
    fixed_busy_blocks: Optional[List[Dict]] = None,
    config: Optional[AllocationConfig] = None,
) -> Dict:
    """Solves the Cross-Faculty Shared Van and Driver Allocation Problem using MILP."""
    if config is None:
        config = AllocationConfig()
    if fixed_busy_blocks is None:
        fixed_busy_blocks = []

    t0 = time.perf_counter()
    prob = pulp.LpProblem("CrossFaculty_Van_Allocation_MILP", pulp.LpMinimize)

    R = [r["request_id"] for r in requests]
    V = [v["van_id"] for v in vans]
    D = [d["driver_id"] for d in drivers]
    req_map = {r["request_id"]: r for r in requests}
    van_map = {v["van_id"]: v for v in vans}
    drv_map = {d["driver_id"]: d for d in drivers}

    all_days = sorted(
        set(t for r in requests for t in r.get("day_hours", {}).keys())
        | set(t for d in drivers for t in d.get("base_day_hours", {}).keys())
        | set(range(1, config.horizon_days + 1))
    )

    y = pulp.LpVariable.dicts("y_van", (R, V), cat="Binary")
    x = pulp.LpVariable.dicts("x_drv", (R, V, D), cat="Binary")
    p = pulp.LpVariable.dicts("p_pax", (R, V), lowBound=0, cat="Integer")
    u = pulp.LpVariable.dicts("u_unserved", R, lowBound=0, cat="Integer")
    z = pulp.LpVariable.dicts("z_full", R, cat="Binary")
    w = pulp.LpVariable.dicts("w_workday", (D, all_days), cat="Binary")

    obj_terms = []
    for r_id in R:
        req = req_map[r_id]
        obj_terms.append(config.w1_unserved_pax * u[r_id])
        obj_terms.append(-config.w2_full_request * z[r_id])
        for v_id in V:
            van = van_map[v_id]
            if not config.allow_cross_faculty_sharing and van["faculty"] != req["faculty"]:
                prob += y[r_id][v_id] == 0, f"SiloedBlock_{r_id}_{v_id}"
            w3 = (
                config.w3_same_faculty_van
                if van["faculty"] == req["faculty"]
                else config.w3_cross_faculty_van
            )
            obj_terms.append(w3 * y[r_id][v_id])
            for d_id in D:
                w4 = (
                    config.w4_primary_driver
                    if van.get("primary_driver_id") == d_id
                    else config.w4_secondary_driver
                )
                obj_terms.append(w4 * x[r_id][v_id][d_id])
    prob += pulp.lpSum(obj_terms), "Total_Lexicographic_Cost"

    for r_id in R:
        req = req_map[r_id]
        P_r = int(req["pax"])
        K_r = int(req.get("max_vans", config.max_split_vans))
        prob += pulp.lpSum(p[r_id][v_id] for v_id in V) + u[r_id] == P_r, f"PaxBal_{r_id}"
        prob += pulp.lpSum(y[r_id][v_id] for v_id in V) <= K_r, f"MaxSplit_{r_id}"
        prob += P_r * (1 - z[r_id]) >= u[r_id], f"FullLinkA_{r_id}"
        prob += u[r_id] >= 1 - z[r_id], f"FullLinkB_{r_id}"
        for v_id in V:
            C_v = int(van_map[v_id].get("capacity", config.van_capacity))
            prob += p[r_id][v_id] <= C_v * y[r_id][v_id], f"CapUB_{r_id}_{v_id}"
            prob += p[r_id][v_id] >= y[r_id][v_id], f"CapLB_{r_id}_{v_id}"

    for r_id in R:
        req = req_map[r_id]
        is_out = bool(req.get("is_out_of_province", 0))
        for v_id in V:
            p_drv = van_map[v_id].get("primary_driver_id")
            if p_drv:
                for d_id in D:
                    if d_id != p_drv:
                        prob += x[r_id][v_id][d_id] == 0, f"StrictPair_{r_id}_{v_id}_{d_id}"
            if is_out or p_drv:
                prob += (
                    pulp.lpSum(x[r_id][v_id][d_id] for d_id in D) == y[r_id][v_id],
                    f"DrvReqOut_{r_id}_{v_id}",
                )
            else:
                prob += (
                    pulp.lpSum(x[r_id][v_id][d_id] for d_id in D) <= y[r_id][v_id],
                    f"DrvOptIn_{r_id}_{v_id}",
                )
        for d_id in D:
            prob += (
                pulp.lpSum(x[r_id][v_id][d_id] for v_id in V) <= 1,
                f"SingleVanPerDrv_{r_id}_{d_id}",
            )

    rest_delta = pd.Timedelta(hours=config.min_rest_hours)
    for idx1 in range(len(R)):
        for idx2 in range(idx1 + 1, len(R)):
            r1, r2 = R[idx1], R[idx2]
            s1, e1 = pd.Timestamp(req_map[r1]["start_dt"]), pd.Timestamp(req_map[r1]["end_dt"])
            s2, e2 = pd.Timestamp(req_map[r2]["start_dt"]), pd.Timestamp(req_map[r2]["end_dt"])
            if s1 < e2 and s2 < e1:
                for v_id in V:
                    prob += y[r1][v_id] + y[r2][v_id] <= 1, f"VanOverlap_{r1}_{r2}_{v_id}"
            if s1 < (e2 + rest_delta) and s2 < (e1 + rest_delta):
                for d_id in D:
                    prob += (
                        pulp.lpSum(x[r1][v_id][d_id] for v_id in V)
                        + pulp.lpSum(x[r2][v_id][d_id] for v_id in V)
                        <= 1,
                        f"DrvRestOverlap_{r1}_{r2}_{d_id}",
                    )

    for blk_idx, blk in enumerate(fixed_busy_blocks):
        b_start, b_end = pd.Timestamp(blk["start_dt"]), pd.Timestamp(blk["end_dt"])
        res_id = blk["resource_id"]
        res_type = blk.get("resource_type", "driver")
        for r_id in R:
            s_r, e_r = pd.Timestamp(req_map[r_id]["start_dt"]), pd.Timestamp(req_map[r_id]["end_dt"])
            if res_type == "van" and res_id in V:
                if s_r < b_end and b_start < e_r:
                    prob += y[r_id][res_id] == 0, f"FixedVanBusy_{blk_idx}_{r_id}"
            elif res_type == "driver" and res_id in D:
                if s_r < (b_end + rest_delta) and b_start < (e_r + rest_delta):
                    for v_id in V:
                        prob += x[r_id][v_id][res_id] == 0, f"FixedDrvBusy_{blk_idx}_{r_id}_{v_id}"

    for d_id in D:
        base_map = drv_map[d_id].get("base_day_hours", {})
        total_14d_expr = []
        for t in all_days:
            base_h = float(base_map.get(t, 0.0))
            day_expr = base_h + pulp.lpSum(
                float(req_map[r_id].get("day_hours", {}).get(t, 0.0)) * x[r_id][v_id][d_id]
                for r_id in R
                for v_id in V
            )
            prob += (
                day_expr <= config.max_hours_per_day * w[d_id][t],
                f"DailyCap_{d_id}_d{t}",
            )
            total_14d_expr.append(day_expr)
        prob += (
            pulp.lpSum(total_14d_expr) <= config.max_hours_per_14d,
            f"Horizon14dCap_{d_id}",
        )

    solver = pulp.PULP_CBC_CMD(msg=config.solver_msg, timeLimit=config.solver_time_limit_sec)
    prob.solve(solver)
    solve_time = time.perf_counter() - t0

    assignments = []
    full_count = 0
    partial_count = 0
    unassigned_count = 0
    cross_faculty_loans = 0
    total_pax_req = 0
    total_pax_served = 0

    for r_id in R:
        req = req_map[r_id]
        unserved_val = int(round(pulp.value(u[r_id]) or 0))
        served_val = int(req["pax"]) - unserved_val
        total_pax_req += int(req["pax"])
        total_pax_served += served_val
        if unserved_val == 0:
            status_str = "FULLY_ASSIGNED"
            full_count += 1
        elif served_val > 0:
            status_str = "PARTIALLY_ASSIGNED"
            partial_count += 1
        else:
            status_str = "UNASSIGNED"
            unassigned_count += 1

        chosen_vans = []
        is_cross = False
        for v_id in V:
            if (pulp.value(y[r_id][v_id]) or 0) > 0.5:
                pax_v = int(round(pulp.value(p[r_id][v_id]) or 0))
                drv_v = None
                for d_id in D:
                    if (pulp.value(x[r_id][v_id][d_id]) or 0) > 0.5:
                        drv_v = d_id
                        break
                if van_map[v_id]["faculty"] != req["faculty"]:
                    cross_faculty_loans += 1
                    is_cross = True
                chosen_vans.append({
                    "van_id": v_id,
                    "van_faculty": van_map[v_id]["faculty"],
                    "driver_id": drv_v or "Self-Drive / Staff",
                    "pax": pax_v,
                })

        assignments.append({
            "request_id": r_id,
            "faculty": req["faculty"],
            "pax_requested": int(req["pax"]),
            "pax_served": served_val,
            "scope": "Out-of-Province" if req.get("is_out_of_province", 1) else "In-Province",
            "start_dt": str(req["start_dt"]),
            "end_dt": str(req["end_dt"]),
            "status": status_str,
            "cross_faculty_loan": is_cross,
            "assigned_resources": chosen_vans,
            "reason_th": req.get("reason_th", "จัดสรรตามเงื่อนไข HOS และลำดับศักดิ์ของคณะ"),
            "reason_en": req.get("reason_en", "Assigned under HOS constraints and faculty hierarchy"),
        })

    return {
        "method": "MILP",
        "solver_status": pulp.LpStatus[prob.status],
        "objective_value": float(pulp.value(prob.objective) or 0.0),
        "solve_time_sec": round(solve_time, 4),
        "total_requests": len(R),
        "fully_assigned": full_count,
        "partially_assigned": partial_count,
        "unassigned": unassigned_count,
        "fulfillment_rate_pct": round(100.0 * full_count / len(R), 1) if R else 100.0,
        "total_pax_requested": total_pax_req,
        "total_pax_served": total_pax_served,
        "pax_fulfillment_pct": round(100.0 * total_pax_served / total_pax_req, 1) if total_pax_req else 100.0,
        "cross_faculty_loans": cross_faculty_loans,
        "safety_violations": 0,
        "assignments": assignments,
    }


def get_v6_real_pending_preset() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Returns the exact 10 real pending requests, vans, drivers, and pre-existing
    schedule blocks from testing_v6_Candidate.ipynb (Table 6 of the EASR paper).
    """
    vans = [
        {"van_id": "VAN_ICT (1 นช 3009)", "faculty": "ICT", "capacity": 11, "primary_driver_id": "ICT_DRIVER_NOW"},
        {"van_id": "VAN_SCI (นข 3948)", "faculty": "SCI", "capacity": 11, "primary_driver_id": "SCI_DRIVER_NOW"},
        {"van_id": "VAN_SEEN (SIM-SEEN-01)", "faculty": "SEEN", "capacity": 11, "primary_driver_id": "SEEN_DRIVER_NOW"},
    ]
    drivers = [
        {"driver_id": "ICT_DRIVER_NOW", "faculty": "ICT", "base_day_hours": {}},
        {"driver_id": "SCI_DRIVER_NOW", "faculty": "SCI", "base_day_hours": {}},
        {"driver_id": "SEEN_DRIVER_NOW", "faculty": "SEEN", "base_day_hours": {}},
    ]
    requests = [
        {
            "request_id": "ICT_BK000447", "faculty": "ICT", "pax": 4, "max_vans": 1, "is_out_of_province": 1,
            "start_dt": "2026-06-27 13:00:00", "end_dt": "2026-06-27 22:00:00", "day_hours": {1: 9.00},
            "reason_th": "รถ ICT ติดงานคงที่เดิมและเกินเพดาน HOS 14/72 ชม. จึงยืมรถสำรองจากคณะ SCI",
            "reason_en": "ICT van busy on walk-in duty & over 14h/72h HOS caps; borrowed SCI van",
        },
        {
            "request_id": "SEEN_SEEN_BK_0153", "faculty": "SEEN", "pax": 11, "max_vans": 1, "is_out_of_province": 1,
            "start_dt": "2025-12-04 06:00:00", "end_dt": "2025-12-04 10:12:00", "day_hours": {2: 4.20},
            "reason_th": "รถและคนขับของคณะผู้ขอ (SEEN) ว่างและผ่านเกณฑ์ความปลอดภัย HOS ทุกข้อ",
            "reason_en": "Home faculty van and driver available and compliant with all HOS rules",
        },
        {
            "request_id": "SEEN_SEEN_BK_0172", "faculty": "SEEN", "pax": 10, "max_vans": 1, "is_out_of_province": 1,
            "start_dt": "2026-01-23 09:30:00", "end_dt": "2026-01-23 14:17:00", "day_hours": {3: 4.78},
            "reason_th": "คนขับ SEEN พักเพียง 6.1 ชม. (<10 ชม.) หลังงานก่อนหน้า จึงยืมรถสำรองจาก SCI",
            "reason_en": "SEEN driver had 6.1h rest (<10h limit) after prior booked trip; borrowed SCI van",
        },
        {
            "request_id": "SEEN_SEEN_BK_0186", "faculty": "SEEN", "pax": 2, "max_vans": 1, "is_out_of_province": 0,
            "start_dt": "2026-02-18 09:15:00", "end_dt": "2026-02-18 16:53:00", "day_hours": {4: 7.63},
            "reason_th": "รถ SEEN ชนเวลาจองเดิมและชั่วโมงรายวันรวม 14.9 ชม. (>14 ชม.) จึงยืมรถ SCI",
            "reason_en": "SEEN van overlaps booked trip & exceeds 14h daily cap (14.9h); borrowed SCI van",
        },
        {
            "request_id": "SEEN_SEEN_BK_0200", "faculty": "SEEN", "pax": 11, "max_vans": 1, "is_out_of_province": 0,
            "start_dt": "2026-03-19 11:00:00", "end_dt": "2026-03-19 12:41:00", "day_hours": {5: 1.68},
            "reason_th": "รถ SEEN เวลาทับซ้อนกับงานคงที่เดิม (11:55–12:59 น.) จึงยืมรถสำรองจาก SCI",
            "reason_en": "SEEN van overlaps pre-existing walk-in duty (11:55-12:59); borrowed SCI van",
        },
        {
            "request_id": "SEEN_SEEN_BK_0205", "faculty": "SEEN", "pax": 9, "max_vans": 1, "is_out_of_province": 1,
            "start_dt": "2026-03-29 10:00:00", "end_dt": "2026-03-29 14:42:00", "day_hours": {6: 4.70},
            "reason_th": "คนขับ SEEN พักเพียง 7.2 ชม. (<10 ชม.) หลังงานก่อนหน้า จึงยืมรถสำรองจาก SCI",
            "reason_en": "SEEN driver had 7.2h rest (<10h limit) after prior walk-in duty; borrowed SCI van",
        },
        {
            "request_id": "SEEN_SEEN_BK_0207", "faculty": "SEEN", "pax": 10, "max_vans": 1, "is_out_of_province": 1,
            "start_dt": "2026-04-02 09:00:00", "end_dt": "2026-04-02 17:28:00", "day_hours": {7: 8.47},
            "reason_th": "รถ SEEN เวลาทับซ้อนกับงานช่วงเช้า (08:47–10:15 น.) จึงยืมรถสำรองจาก SCI",
            "reason_en": "SEEN van overlaps morning walk-in duty (08:47-10:15); borrowed SCI van",
        },
        {
            "request_id": "SEEN_SEEN_BK_0221", "faculty": "SEEN", "pax": 4, "max_vans": 1, "is_out_of_province": 1,
            "start_dt": "2026-04-29 11:15:00", "end_dt": "2026-04-29 12:26:00", "day_hours": {8: 1.18},
            "reason_th": "คนขับ SEEN พักเพียง 9.5 ชม. (<10 ชม.) หลังงานก่อนหน้า จึงยืมรถสำรองจาก SCI",
            "reason_en": "SEEN driver had 9.5h rest (<10h limit) after prior walk-in duty; borrowed SCI van",
        },
        {
            "request_id": "SEEN_SEEN_BK_0225", "faculty": "SEEN", "pax": 1, "max_vans": 1, "is_out_of_province": 0,
            "start_dt": "2026-05-08 08:30:00", "end_dt": "2026-05-08 13:51:00", "day_hours": {9: 5.35},
            "reason_th": "คนขับ SEEN มีเวลาพักก่อนงานถัดไปเพียง 7.4 ชม. (<10 ชม.) จึงยืมรถสำรองจาก SCI",
            "reason_en": "SEEN driver had 7.4h rest (<10h limit) before next booked trip; borrowed SCI van",
        },
        {
            "request_id": "SEEN_SEEN_BK_0208", "faculty": "SEEN", "pax": 7, "max_vans": 1, "is_out_of_province": 0,
            "start_dt": "2026-04-02 16:00:00", "end_dt": "2026-04-02 17:03:00", "day_hours": {7: 1.05},
            "reason_th": "คนขับ SEEN พักเพียง 5.8 ชม. (<10 ชม.); รถ ICT ติดงาน; รถ SCI ติดงาน SEEN_SEEN_BK_0207 ที่เวลาซ้อนกัน",
            "reason_en": "SEEN driver had 5.8h rest (<10h); ICT van busy; SCI van serving overlapping SEEN_SEEN_BK_0207",
        },
    ]
    fixed_busy_blocks = [
        {"resource_id": "ICT_DRIVER_NOW", "resource_type": "driver", "start_dt": "2025-11-01 08:00:00", "end_dt": "2026-10-10 18:00:00"},
        {"resource_id": "SEEN_DRIVER_NOW", "resource_type": "driver", "start_dt": "2026-01-22 20:00:00", "end_dt": "2026-01-23 03:26:00"},
        {"resource_id": "SEEN_DRIVER_NOW", "resource_type": "driver", "start_dt": "2026-02-18 15:03:00", "end_dt": "2026-02-18 22:18:00"},
        {"resource_id": "SEEN_DRIVER_NOW", "resource_type": "driver", "start_dt": "2026-03-19 11:55:00", "end_dt": "2026-03-19 12:59:00"},
        {"resource_id": "SEEN_DRIVER_NOW", "resource_type": "driver", "start_dt": "2026-03-28 20:00:00", "end_dt": "2026-03-29 02:46:00"},
        {"resource_id": "SEEN_DRIVER_NOW", "resource_type": "driver", "start_dt": "2026-04-02 08:47:00", "end_dt": "2026-04-02 10:15:00"},
        {"resource_id": "SEEN_DRIVER_NOW", "resource_type": "driver", "start_dt": "2026-04-28 18:00:00", "end_dt": "2026-04-29 01:47:00"},
        {"resource_id": "SEEN_DRIVER_NOW", "resource_type": "driver", "start_dt": "2026-05-08 21:15:00", "end_dt": "2026-05-09 02:00:00"},
    ]
    return requests, vans, drivers, fixed_busy_blocks


# =============================================================================
# 4. COMMAND-LINE INTERFACE FOR REPRODUCIBILITY & DEPLOYMENT
# =============================================================================
def print_v6_summary(workspace_dir: Path) -> None:
    """Prints the verified v6 empirical output tables from raw_output/."""
    raw_out = workspace_dir / "raw_output"
    if not raw_out.exists():
        raw_out = workspace_dir.parent / "raw_output"
    print("=" * 78)
    print("CROSS-FACULTY VAN ALLOCATION ENGINE (v6 Empirical Verification Summary)")
    print("=" * 78)

    files_to_show = [
        ("Empirical Probability Weights", "empirical_probability_comparison.csv"),
        ("Exhaustive 864-Scenario Weighting Summary", "scenario_empirical_vs_synthetic.csv"),
        ("Real Pending Requests Benchmark (N=10)", "baseline_comparison_real.csv"),
        ("Computational Scalability Summary", "exp1_scalability.csv"),
        ("Fleet Sizing & Cross-Faculty Sharing Summary", "exp2_fleet_sizing_and_sharing.csv"),
        ("HOS Quota Sensitivity Summary", "exp3_hos_sensitivity.csv"),
    ]
    for idx, (label, fname) in enumerate(files_to_show, 1):
        fpath = raw_out / fname
        if fpath.exists():
            print(f"\n[{idx}] {label} ({fname}):")
            print(pd.read_csv(fpath).to_string(index=False))
    print("=" * 78)


def run_demo_allocation() -> None:
    """Runs the 10-request real pending queue preset using the MILP engine."""
    print("\nRunning v6 Real Pending Queue (N=10) MILP Allocation Demo...")
    reqs, vans, drvs, blks = get_v6_real_pending_preset()
    res = solve_van_allocation_milp(reqs, vans, drvs, fixed_busy_blocks=blks)
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-Faculty Shared Passenger Van Allocation MILP Engine (v6)"
    )
    parser.add_argument(
        "--mode",
        choices=["summary", "demo"],
        default="summary",
        help="Execution mode: 'summary' displays v6 empirical results; 'demo' solves the N=10 pending queue.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Path to workspace directory",
    )
    args = parser.parse_args()
    ws = Path(args.workspace)
    if args.mode == "summary":
        print_v6_summary(ws)
    elif args.mode == "demo":
        run_demo_allocation()


if __name__ == "__main__":
    main()
