exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())
exec(open('walk_sim_vnoid.py').read())

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import walking_control


def vec(row, prefix, value):
    arr = np.asarray(value if value is not None else [0.0, 0.0, 0.0], dtype=float).reshape(-1)
    row[f"{prefix}_x"] = f"{arr[0]:.12g}"
    row[f"{prefix}_y"] = f"{arr[1]:.12g}"
    row[f"{prefix}_z"] = f"{arr[2]:.12g}"


def base_row(kind, stage, wc, index=-1):
    return {
        "source": "python",
        "kind": kind,
        "stage": stage,
        "count": wc.timer.count if wc.timer is not None else 0,
        "time": f"{wc.timer.time:.12g}" if wc.timer is not None else "0",
        "index": index,
        "side": "",
        "stepping": "",
        "contact0": int(wc.feet[0].contact_ref) if wc.feet else "",
        "contact1": int(wc.feet[1].contact_ref) if wc.feet else "",
    }


def log_state(writer, stage, wc):
    row = base_row("state", stage, wc)
    vec(row, "rfoot", wc.feet[0].pos_ref)
    vec(row, "lfoot", wc.feet[1].pos_ref)
    vec(row, "com", wc.centroid.com_pos_ref)
    vec(row, "dcm_ref", wc.centroid.dcm_ref)
    vec(row, "dcm_target", wc.centroid.dcm_target)
    vec(row, "zmp_ref", wc.centroid.zmp_ref)
    vec(row, "zmp_target", wc.centroid.zmp_target)
    vec(row, "force0", wc.feet[0].force)
    vec(row, "force1", wc.feet[1].force)
    writer.writerow(row)


def log_footsteps(writer, stage, wc):
    for idx, step in enumerate(wc.footstep.steps):
        row = base_row("footstep", stage, wc, idx)
        row["side"] = step.side
        row["stepping"] = int(step.stepping)
        vec(row, "rfoot", step.foot_pos[0])
        vec(row, "lfoot", step.foot_pos[1])
        vec(row, "com", [0.0, 0.0, 0.0])
        vec(row, "dcm_ref", step.dcm)
        vec(row, "dcm_target", [0.0, 0.0, 0.0])
        vec(row, "zmp_ref", step.zmp)
        vec(row, "zmp_target", [0.0, 0.0, 0.0])
        vec(row, "force0", [0.0, 0.0, 0.0])
        vec(row, "force1", [0.0, 0.0, 0.0])
        writer.writerow(row)


def fmt(value):
    arr = np.asarray(value if value is not None else [0.0, 0.0, 0.0], dtype=float).reshape(-1)
    return f"({arr[0]: .6f}, {arr[1]: .6f}, {arr[2]: .6f})"


def print_footsteps(label, wc):
    print(f"\n===== PYTHON {label} FOOTSTEPS =====")
    for idx, step in enumerate(wc.footstep.steps):
        print(
            f"step[{idx}] side={step.side} stepping={int(step.stepping)} "
            f"duration={step.duration:.6f} tbegin={step.tbegin:.6f}"
        )
        print(f"  rfoot={fmt(step.foot_pos[0])} angle={fmt(step.foot_angle[0])}")
        print(f"  lfoot={fmt(step.foot_pos[1])} angle={fmt(step.foot_angle[1])}")
        print(f"  zmp  ={fmt(step.zmp)}")
        print(f"  dcm  ={fmt(step.dcm)}")


def print_state(label, wc):
    print(f"\n===== PYTHON {label} STATE count={wc.timer.count} time={wc.timer.time:.6f} =====")
    print(f"  rfoot_ref ={fmt(wc.feet[0].pos_ref)} contact_ref={int(wc.feet[0].contact_ref)}")
    print(f"  lfoot_ref ={fmt(wc.feet[1].pos_ref)} contact_ref={int(wc.feet[1].contact_ref)}")
    print(f"  com_ref   ={fmt(wc.centroid.com_pos_ref)}")
    print(f"  dcm_ref   ={fmt(wc.centroid.dcm_ref)}")
    print(f"  dcm_target={fmt(wc.centroid.dcm_target)}")
    print(f"  zmp_ref   ={fmt(wc.centroid.zmp_ref)}")
    print(f"  zmp_target={fmt(wc.centroid.zmp_target)}")
    print(f"  force[0]  ={fmt(wc.feets[0].force) if hasattr(wc, 'feets') else fmt(wc.feet[0].force)}")
    print(f"  force[1]  ={fmt(wc.feets[1].force) if hasattr(wc, 'feets') else fmt(wc.feet[1].force)}")


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    # vnoid_python_test.cpp と完全に同じシナリオにするため、Param を明示的に揃える。
    from footstep_planner import Param
    param = Param()
    param.total_mass = 50.0
    param.com_height = 0.70
    param.gravity = 9.8
    param.trunk_mass = 24.0
    param.trunk_com = np.array([0.0, 0.0, 0.166])
    param.zmp_min = np.array([-0.1, -0.05, -0.1])
    param.zmp_max = np.array([0.1, 0.05, 0.1])
    # nominal_inertia は C++テスト側で明示設定されていないため Param() デフォルト (20,20,5) のまま
    param.Init()

    wc = walking_control.WalkingControl(dt=0.01)
    wc.setup_controller(
        param=param,
        stride=0.1,
        spacing=0.2,
        duration=0.5,
        stepSize=4,
    )
    wc.stepping_controller.debug = 0
    wc.use_cpp_stabilizer = True
    wc.no_dcm_gain = False
    wc.no_dcm_derivative = False

    print_footsteps("AFTER_GENERATE_DCM", wc)
    print_state("INITIAL", wc)

    for _ in range(steps):
        wc.step_simulation()
        print_state("AFTER_STEP_SIMULATION", wc)


if __name__ == "__main__":
    main()