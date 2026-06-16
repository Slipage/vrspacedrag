#!/usr/bin/env python3
"""Terminal-only OpenVR space drag prototype."""

from __future__ import annotations

import math
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from typing import Optional

try:
    import openvr
except ImportError:
    print(
        "Missing Python OpenVR bindings. Install them with: python3 -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)


REAL_HEIGHT_METERS = 1.778  # 5 ft 10 in
DRAG_SPEED_MULTIPLIER = 3.0
TARGET_HZ = 90.0
FRAME_INTERVAL_SECONDS = 1.0 / TARGET_HZ
DEBUG_INTERVAL_SECONDS = 0.25
MIN_VALID_HMD_HEIGHT_METERS = 0.5
DEADZONE_METERS = 0.003
MAX_DELTA_METERS = 0.150
SMOOTHING_HZ = 18.0
FLING_MULTIPLIER = 8.0
FLING_DECEL_HZ = 4.0
GRAVITY_SMOOTH_HZ = 8.0
BODY_EMULATION_HZ = 4.0
TRACKING_ORIGIN = openvr.TrackingUniverseStanding
JOYSTICK_AXIS_TYPE = openvr.k_eControllerAxis_Joystick
AXIS_PROPERTIES = (
    openvr.Prop_Axis0Type_Int32,
    openvr.Prop_Axis1Type_Int32,
    openvr.Prop_Axis2Type_Int32,
    openvr.Prop_Axis3Type_Int32,
    openvr.Prop_Axis4Type_Int32,
)


@dataclass(slots=True)
class ControllerBinding:
    device_index: int
    axis_index: int
    button_id: int
    model: str


@dataclass(slots=True)
class RuntimeState:
    scale_factor: float = 1.0
    gravity_enabled: bool = False
    gravity_target: float = 0.0
    gravity_current: float = 0.0
    drag_active: bool = False
    hmd_height: float = 0.0
    floor_offset_y: float = 0.0
    movement_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    current_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    smoothed_delta: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fling_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    body_emulation_active: bool = False
    body_waist_offset: float = 0.0
    body_foot_offset: float = 0.0
    right_previous_position: Optional[tuple[float, float, float]] = None
    last_left_button_pressed: bool = False
    last_debug_time: float = 0.0


class TerminalKeyboard:
    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._previous_settings: Optional[list] = None

    def __enter__(self) -> "TerminalKeyboard":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._previous_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fd is not None and self._previous_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._previous_settings)

    def poll_key(self) -> Optional[str]:
        if self._fd is None:
            return None
        ready, _, _ = select.select([self._fd], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1)


class OpenVrRuntime:
    def __init__(self, max_retries: int = 30, retry_delay: float = 0.1) -> None:
        self.system = openvr.init(openvr.VRApplication_Background)
        self.chaperone_setup = openvr.VRChaperoneSetup()
        self.poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        self.chaperone_setup.revertWorkingCopy()
        self.baseline_matrix = self._capture_baseline()
        
        self.left_controller = None
        self.right_controller = None
        
        for attempt in range(max_retries):
            self.poll_poses()
            self.left_controller = self._find_controller_binding(
                openvr.TrackedControllerRole_LeftHand
            )
            self.right_controller = self._find_controller_binding(
                openvr.TrackedControllerRole_RightHand
            )
            if self.left_controller is not None and self.right_controller is not None:
                break
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    def shutdown(self) -> None:
        try:
            self.chaperone_setup.revertWorkingCopy()
            self.chaperone_setup.hideWorkingSetPreview()
        except Exception:
            pass
        openvr.shutdown()

    def poll_poses(self):
        return self.system.getDeviceToAbsoluteTrackingPose(
            TRACKING_ORIGIN, 0.0, self.poses
        )

    def read_hmd_height(self, poses) -> Optional[float]:
        position = pose_to_position(poses[openvr.k_unTrackedDeviceIndex_Hmd])
        if position is None:
            return None
        return position[1]

    def read_controller_snapshot(
        self, binding: Optional[ControllerBinding], poses
    ) -> tuple[bool, Optional[tuple[float, float, float]]]:
        if binding is None:
            return False, None

        pose = poses[binding.device_index]
        position = pose_to_position(pose)

        try:
            valid, controller_state = self.system.getControllerState(binding.device_index)
        except Exception:
            return False, position

        if not valid:
            return False, position

        pressed = bool(controller_state.ulButtonPressed & (1 << binding.button_id))
        return pressed, position

    def apply_offset(self, offset: list[float], height_emulation: float = 0.0) -> None:
        matrix = openvr.HmdMatrix34_t()
        for row in range(3):
            for column in range(4):
                matrix.m[row][column] = self.baseline_matrix[row][column]

        for row in range(3):
            matrix.m[row][3] = (
                self.baseline_matrix[row][3]
                + self.baseline_matrix[row][0] * offset[0]
                + self.baseline_matrix[row][1] * (offset[1] + height_emulation)
                + self.baseline_matrix[row][2] * offset[2]
            )

        self.chaperone_setup.setWorkingStandingZeroPoseToRawTrackingPose(matrix)
        self.chaperone_setup.showWorkingSetPreview()

    def _capture_baseline(self) -> list[list[float]]:
        _, matrix = self.chaperone_setup.getWorkingStandingZeroPoseToRawTrackingPose()
        return [[float(matrix.m[row][column]) for column in range(4)] for row in range(3)]

    def _find_controller_binding(self, role: int) -> Optional[ControllerBinding]:
        device_index = self.system.getTrackedDeviceIndexForControllerRole(role)
        if device_index == openvr.k_unTrackedDeviceIndexInvalid:
            return None

        axis_index = None
        for index, property_id in enumerate(AXIS_PROPERTIES):
            try:
                axis_type = self.system.getInt32TrackedDeviceProperty(
                    device_index, property_id
                )
            except Exception:
                continue
            if axis_type == JOYSTICK_AXIS_TYPE:
                axis_index = index
                break

        if axis_index is None:
            return None

        try:
            model = self.system.getStringTrackedDeviceProperty(
                device_index, openvr.Prop_ModelNumber_String
            )
        except Exception:
            model = "unknown"

        return ControllerBinding(
            device_index=device_index,
            axis_index=axis_index,
            button_id=int(openvr.k_EButton_Axis0) + axis_index,
            model=model,
        )

    def refresh_controller_binding(self, role: int) -> Optional[ControllerBinding]:
        binding = self._find_controller_binding(role)
        if role == openvr.TrackedControllerRole_LeftHand:
            self.left_controller = binding
        elif role == openvr.TrackedControllerRole_RightHand:
            self.right_controller = binding
        return binding


def calibrate_height(state: RuntimeState, current_hmd_height: float) -> None:
    if current_hmd_height < MIN_VALID_HMD_HEIGHT_METERS:
        raise RuntimeError(
            "HMD height is too small to calibrate. Stand upright and press R to try again."
        )

    state.hmd_height = current_hmd_height
    state.scale_factor = REAL_HEIGHT_METERS / current_hmd_height
    state.floor_offset_y += state.movement_offset[1]
    state.movement_offset[1] = 0.0
    state.current_offset[1] = state.floor_offset_y
    state.right_previous_position = None
    state.smoothed_delta[0] = 0.0
    state.smoothed_delta[1] = 0.0
    state.smoothed_delta[2] = 0.0
    state.fling_velocity[0] = 0.0
    state.fling_velocity[1] = 0.0
    state.fling_velocity[2] = 0.0
    state.drag_active = False
    
    state.body_waist_offset = -(current_hmd_height - 0.95)
    state.body_foot_offset = -(current_hmd_height - 0.85)
    state.body_emulation_active = True


def apply_space_drag(
    state: RuntimeState,
    current_position: Optional[tuple[float, float, float]],
    drag_button_pressed: bool,
    dt_seconds: float,
) -> bool:
    if not drag_button_pressed or current_position is None:
        was_active = state.drag_active
        state.drag_active = False
        
        if was_active and state.gravity_current < 0.5 and vector_length(state.smoothed_delta) > DEADZONE_METERS:
            for i in range(3):
                state.fling_velocity[i] = state.smoothed_delta[i] * FLING_MULTIPLIER
        
        state.right_previous_position = None
        state.smoothed_delta[0] = 0.0
        state.smoothed_delta[1] = 0.0
        state.smoothed_delta[2] = 0.0
        return False

    if not state.drag_active or state.right_previous_position is None:
        state.drag_active = True
        state.right_previous_position = current_position
        return False

    previous = state.right_previous_position
    raw_delta = [
        (current_position[0] - previous[0])
        * state.scale_factor
        * DRAG_SPEED_MULTIPLIER,
        (current_position[1] - previous[1])
        * state.scale_factor
        * DRAG_SPEED_MULTIPLIER,
        (current_position[2] - previous[2])
        * state.scale_factor
        * DRAG_SPEED_MULTIPLIER,
    ]
    state.right_previous_position = current_position

    if state.gravity_current > 0.5:
        raw_delta[1] = 0.0

    raw_length = vector_length(raw_delta)
    if raw_length > MAX_DELTA_METERS:
        scale = MAX_DELTA_METERS / raw_length
        raw_delta[0] *= scale
        raw_delta[1] *= scale
        raw_delta[2] *= scale
        raw_length = MAX_DELTA_METERS

    if raw_length < DEADZONE_METERS:
        raw_delta[0] = 0.0
        raw_delta[1] = 0.0
        raw_delta[2] = 0.0

    alpha = 1.0 - math.exp(-SMOOTHING_HZ * dt_seconds)
    for index in range(3):
        state.smoothed_delta[index] += (
            raw_delta[index] - state.smoothed_delta[index]
        ) * alpha

    if raw_length < DEADZONE_METERS:
        if vector_length(state.smoothed_delta) < DEADZONE_METERS:
            state.smoothed_delta[0] = 0.0
            state.smoothed_delta[1] = 0.0
            state.smoothed_delta[2] = 0.0
        return False

    if vector_length(state.smoothed_delta) < DEADZONE_METERS:
        return False

    # OpenVR standing-zero offsets already operate in the inverse-world direction.
    state.movement_offset[0] += state.smoothed_delta[0]
    state.movement_offset[1] += state.smoothed_delta[1]
    state.movement_offset[2] += state.smoothed_delta[2]
    return True


def apply_fling(state: RuntimeState, dt_seconds: float) -> bool:
    if vector_length(state.fling_velocity) < 1e-6:
        return False
    
    alpha = 1.0 - math.exp(-FLING_DECEL_HZ * dt_seconds)
    for i in range(3):
        state.fling_velocity[i] *= (1.0 - alpha)
    
    if vector_length(state.fling_velocity) < 1e-6:
        state.fling_velocity[0] = 0.0
        state.fling_velocity[1] = 0.0
        state.fling_velocity[2] = 0.0
        return False
    
    for i in range(3):
        state.movement_offset[i] += state.fling_velocity[i] * dt_seconds
    return True


def toggle_gravity(state: RuntimeState) -> None:
    state.gravity_enabled = not state.gravity_enabled
    state.gravity_target = 1.0 if state.gravity_enabled else 0.0
    if state.gravity_enabled:
        state.movement_offset[1] = 0.0
        state.smoothed_delta[1] = 0.0
        state.fling_velocity[1] = 0.0


def update_transform(runtime: OpenVrRuntime, state: RuntimeState) -> bool:
    target_y = state.floor_offset_y
    if state.gravity_current < 0.5:
        target_y += state.movement_offset[1]

    next_offset = [
        state.movement_offset[0],
        target_y,
        state.movement_offset[2],
    ]

    if offsets_equal(next_offset, state.current_offset) and abs(state.height_emulation_current - state.height_emulation_target) < 1e-4:
        return False

    state.current_offset[0] = next_offset[0]
    state.current_offset[1] = next_offset[1]
    state.current_offset[2] = next_offset[2]
    runtime.apply_offset(state.current_offset, state.height_emulation_current)
    return True


def update_gravity(state: RuntimeState, dt_seconds: float) -> bool:
    if abs(state.gravity_current - state.gravity_target) < 1e-4:
        state.gravity_current = state.gravity_target
        return False
    
    alpha = 1.0 - math.exp(-GRAVITY_SMOOTH_HZ * dt_seconds)
    state.gravity_current += (state.gravity_target - state.gravity_current) * alpha
    return True


def offsets_equal(a: list[float], b: list[float]) -> bool:
    return (
        abs(a[0] - b[0]) < 1e-6
        and abs(a[1] - b[1]) < 1e-6
        and abs(a[2] - b[2]) < 1e-6
    )


def vector_length(values: list[float]) -> float:
    return math.sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])


def pose_to_position(pose) -> Optional[tuple[float, float, float]]:
    if (
        not pose.bPoseIsValid
        or not pose.bDeviceIsConnected
        or pose.eTrackingResult != openvr.TrackingResult_Running_OK
    ):
        return None

    matrix = pose.mDeviceToAbsoluteTracking.m
    return (
        float(matrix[0][3]),
        float(matrix[1][3]),
        float(matrix[2][3]),
    )


def print_status(state: RuntimeState) -> None:
    body_info = ""
    if state.body_emulation_active:
        body_info = f" | waist: {state.body_waist_offset:+.2f}m"
    line = (
        "\r"
        f"HMD height: {state.hmd_height:0.3f} m | "
        f"scale: {state.scale_factor:0.4f} | "
        f"drag speed: x{DRAG_SPEED_MULTIPLIER:0.1f} | "
        f"gravity: {'ON' if state.gravity_enabled else 'OFF'} | "
        f"offset: ({state.current_offset[0]:+0.3f}, "
        f"{state.current_offset[1]:+0.3f}, "
        f"{state.current_offset[2]:+0.3f})"
        f"{body_info}"
    )
    sys.stdout.write(line.ljust(150))
    sys.stdout.flush()


def main() -> int:
    runtime = OpenVrRuntime()
    state = RuntimeState()

    try:
        initial_poses = runtime.poll_poses()
        initial_height = runtime.read_hmd_height(initial_poses)
        if initial_height is None:
            raise RuntimeError("Could not read a valid HMD pose from OpenVR.")
        calibrate_height(state, initial_height)
        update_transform(runtime, state)

        print("OpenVR space drag running")
        print(
            "Controls: hold right joystick click to drag, left joystick click to toggle gravity,"
            " press C to refresh controllers, press R to recalibrate, press Q to quit."
        )
        if runtime.left_controller is not None:
            print(f"Left controller: {runtime.left_controller.model}")
        else:
            print("Left controller: joystick click binding not found")
        if runtime.right_controller is not None:
            print(f"Right controller: {runtime.right_controller.model}")
        else:
            print("Right controller: joystick click binding not found")
        state.last_debug_time = time.perf_counter()
        print_status(state)

        with TerminalKeyboard() as keyboard:
            last_frame_time = time.perf_counter()
            while True:
                frame_start = time.perf_counter()
                dt_seconds = max(1e-4, min(0.050, frame_start - last_frame_time))
                last_frame_time = frame_start
                key = keyboard.poll_key()
                if key in {"q", "Q"}:
                    break

                gravity_changed = update_gravity(state, dt_seconds)
                
                fling_applied = False
                if not state.drag_active:
                    fling_applied = apply_fling(state, dt_seconds)

                poses = runtime.poll_poses()
                hmd_height = runtime.read_hmd_height(poses)
                if hmd_height is not None:
                    state.hmd_height = hmd_height

                if key in {"r", "R"} and hmd_height is not None:
                    calibrate_height(state, hmd_height)
                    update_transform(runtime, state)
                    print("\nRecalibrated floor level and height baseline.")

                if key in {"c", "C"}:
                    runtime.refresh_controller_binding(openvr.TrackedControllerRole_LeftHand)
                    runtime.refresh_controller_binding(openvr.TrackedControllerRole_RightHand)
                    print("\nRefreshed controller bindings.")
                    if runtime.left_controller is not None:
                        print(f"Left controller: {runtime.left_controller.model}")
                    else:
                        print("Left controller: not detected")
                    if runtime.right_controller is not None:
                        print(f"Right controller: {runtime.right_controller.model}")
                    else:
                        print("Right controller: not detected")

                poses = runtime.poll_poses()
                hmd_height = runtime.read_hmd_height(poses)
                if hmd_height is not None:
                    state.hmd_height = hmd_height

                if key in {"r", "R"} and hmd_height is not None:
                    calibrate_height(state, hmd_height)
                    update_transform(runtime, state)
                    print("\nRecalibrated floor level and height baseline.")

                left_pressed, _ = runtime.read_controller_snapshot(
                    runtime.left_controller, poses
                )
                if left_pressed and not state.last_left_button_pressed:
                    toggle_gravity(state)
                    print(
                        f"\nGravity {'enabled' if state.gravity_enabled else 'disabled'}."
                    )
                state.last_left_button_pressed = left_pressed

                right_pressed, right_position = runtime.read_controller_snapshot(
                    runtime.right_controller, poses
                )
                moved = apply_space_drag(
                    state,
                    current_position=right_position,
                    drag_button_pressed=right_pressed,
                    dt_seconds=dt_seconds,
                )
                
                if gravity_changed or fling_applied or moved:
                    update_transform(runtime, state)

                now = time.perf_counter()
                if now - state.last_debug_time >= DEBUG_INTERVAL_SECONDS:
                    state.last_debug_time = now
                    print_status(state)

                elapsed = time.perf_counter() - frame_start
                sleep_for = FRAME_INTERVAL_SECONDS - elapsed
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
    finally:
        print()
        runtime.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
