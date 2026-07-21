"""uav_wm.planning — danger-aware CEM planner (NavPlanner) + offline eval.

NavPlanner / DangerPlanner / imagined_danger live in :mod:`uav_wm.planning.cem_planner`.
They reuse swm's CEMSolver + the frozen LeWM world model + danger head; the host-side
socket server (`scripts/live/planner_server.py`) imports them to plan batched actions.
"""
