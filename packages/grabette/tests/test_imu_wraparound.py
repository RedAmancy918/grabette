"""IMU capture + driver tests (hardware-free).

ImuCapture must keep timestamps monotonic across a hardware-counter wrap,
whatever the counter width: the BMI088's 24-bit SENSORTIME (39.0625 us/tick,
wraps ~655 s) and the LSM6DSV16X's 32-bit TIMESTAMP (~21.75 us/tick). Any
recording can start just before a wrap, so samples after it must not jump back
and get dropped by the cts >= 0 filter. Also covers the LSM6DSV16X register
math and ImuDriver conformance via fake I2C.
"""
import struct
import time

import pytest

from grabette.hardware.bmi088 import BMI088, SENSORTIME_RESOLUTION_US
from grabette.hardware.imu import ImuCapture, ImuDriver, create_imu_driver
from grabette.hardware.lsm6dsv16x import (
    LSM6DSV16X, TIMESTAMP_RESOLUTION_US, WHO_AM_I, OUTX_L_A, OUTX_L_G, TIMESTAMP0,
)
from grabette.hardware.sync import SyncManager


class FakeCounterDriver:
    """ImuDriver whose counter tracks real time and starts just before a wrap."""

    def __init__(self, counter_bits: int, res_us: float, ticks_before_wrap: int):
        self.COUNTER_BITS = counter_bits
        self.TIMESTAMP_RESOLUTION_US = res_us
        self._wrap = 1 << counter_bits
        self._t0 = time.monotonic()
        self._start = self._wrap - ticks_before_wrap

    def init(self) -> None:
        pass

    def read_timestamp(self) -> int:
        elapsed = int((time.monotonic() - self._t0) * 1e6 / self.TIMESTAMP_RESOLUTION_US)
        return (self._start + elapsed) % self._wrap

    def read_accel_with_time(self):
        return (0.0, 9.81, 0.0), self.read_timestamp()

    def read_gyro(self):
        return (0.0, 0.0, 0.0)

    def read_accel(self):
        return (0.0, 9.81, 0.0)

    def deinit(self) -> None:
        pass


@pytest.mark.parametrize("bits,res_us", [(24, SENSORTIME_RESOLUTION_US), (32, TIMESTAMP_RESOLUTION_US)])
def test_timestamps_monotonic_across_wrap(bits, res_us):
    sync = SyncManager()
    sync.start()
    # Start ~100 ms before the wrap; capture runs ~300 ms so it crosses it.
    driver = FakeCounterDriver(bits, res_us, ticks_before_wrap=int(0.1e6 / res_us))
    cap = ImuCapture(sync, driver=driver, sample_rate_hz=200)

    cap.start_capture()
    time.sleep(0.3)
    samples = cap.stop()

    cts = [s["cts"] for s in samples.accel]
    # 200 Hz for ~0.3 s; a bad unwrap would keep only the ~20 pre-wrap samples.
    assert len(cts) > 40
    assert all(b > a for a, b in zip(cts, cts[1:]))
    # Timestamps stay on the sync clock: the span matches wall time.
    assert 250 < cts[-1] - cts[0] < 400


class FakeI2C:
    """Minimal adafruit-style I2C returning canned register reads."""

    def __init__(self, regs: dict[int, bytes]):
        self.regs = regs
        self.writes: list[tuple[int, bytes]] = []

    def try_lock(self) -> bool:
        return True

    def unlock(self) -> None:
        pass

    def writeto(self, addr, data) -> None:
        self.writes.append((data[0], bytes(data[1:])))

    def writeto_then_readfrom(self, addr, out, inbuf) -> None:
        data = self.regs.get(out[0], bytes(len(inbuf)))
        inbuf[:] = data[: len(inbuf)].ljust(len(inbuf), b"\x00")

    def deinit(self) -> None:
        pass


def test_lsm6dsv16x_scaling_and_timestamp():
    regs = {
        WHO_AM_I: bytes([0x70]),
        OUTX_L_A: struct.pack("<hhh", 1000, 0, -2000),
        OUTX_L_G: struct.pack("<hhh", 100, -100, 0),
        TIMESTAMP0: bytes([0x10, 0x00, 0x00, 0x00]),
    }
    dev = LSM6DSV16X(FakeI2C(regs))
    dev.init()  # WHO_AM_I matches -> no raise

    ax, ay, az = dev.read_accel()
    assert ay == 0.0
    assert ax == pytest.approx(1000 * 0.244e-3 * 9.80665, rel=1e-6)   # 8 g
    assert az == pytest.approx(-2000 * 0.244e-3 * 9.80665, rel=1e-6)

    gx, gy, gz = dev.read_gyro()
    assert gx == pytest.approx(100 * 70e-3 * 3.14159265359 / 180.0, rel=1e-6)  # 2000 dps
    assert gz == 0.0

    assert dev.read_timestamp() == 0x10  # little-endian 32-bit assembly


def test_lsm6dsv16x_wrong_whoami_raises():
    dev = LSM6DSV16X(FakeI2C({WHO_AM_I: bytes([0x6A])}))
    with pytest.raises(RuntimeError, match="WHO_AM_I"):
        dev.init()


def test_drivers_conform_to_protocol():
    assert isinstance(LSM6DSV16X(FakeI2C({})), ImuDriver)
    assert isinstance(BMI088(FakeI2C({})), ImuDriver)


def test_create_imu_driver_selects_sensor_and_odr():
    from grabette.hardware.lsm6dsv16x import ODR_120Hz, ODR_240Hz

    assert isinstance(create_imu_driver("lsm6dsv16x", FakeI2C({}), 240), LSM6DSV16X)
    assert create_imu_driver("lsm6dsv16x", FakeI2C({}), 240)._odr == ODR_240Hz
    assert create_imu_driver("lsm6dsv16x", FakeI2C({}), 100)._odr == ODR_120Hz
    assert isinstance(create_imu_driver("bmi088", FakeI2C({}), 200), BMI088)
    with pytest.raises(ValueError, match="unknown IMU sensor"):
        create_imu_driver("nope", FakeI2C({}), 200)
