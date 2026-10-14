"""The depth camera stays on under a fleet lease, and powers down once it expires."""
import asyncio
import types

from grabette.backend import rpi


def _warm_backend(keepalive_s=0.05):
    b = rpi.RpiBackend()
    b._oakd_keepalive_s = keepalive_s
    b._oakd = types.SimpleNamespace(is_initialized=True)
    b._oakd_auto_enabled = True
    powered_down = []

    async def _set_enabled(on):
        powered_down.append(not on)
        b._oakd = None

    b.set_oakd_enabled = _set_enabled
    return b, powered_down


def test_without_a_hold_the_keepalive_powers_the_camera_down():
    b, powered_down = _warm_backend()

    asyncio.run(b._oakd_keepalive_powerdown())

    assert powered_down == [True]


def test_a_live_hold_keeps_the_camera_on_past_the_keepalive():
    b, powered_down = _warm_backend(keepalive_s=0.01)

    async def scenario():
        b.hold_depth_camera(10)
        task = asyncio.create_task(b._oakd_keepalive_powerdown())
        await asyncio.sleep(0.1)
        done = task.done()
        task.cancel()
        return done

    assert asyncio.run(scenario()) is False
    assert powered_down == []


def test_an_expired_hold_hands_power_back_to_the_keepalive():
    # Nobody renews the lease (the fleet went away): the camera must still go off.
    b, powered_down = _warm_backend(keepalive_s=0.01)
    b.hold_depth_camera(0.05)

    asyncio.run(asyncio.wait_for(b._oakd_keepalive_powerdown(), timeout=2))

    assert powered_down == [True]


def test_a_renewed_hold_is_seen_by_a_powerdown_already_waiting():
    b, powered_down = _warm_backend(keepalive_s=0.01)

    async def scenario():
        b.hold_depth_camera(0.05)
        task = asyncio.create_task(b._oakd_keepalive_powerdown())
        await asyncio.sleep(0.03)
        b.hold_depth_camera(10)
        await asyncio.sleep(0.1)
        done = task.done()
        task.cancel()
        return done

    assert asyncio.run(scenario()) is False
    assert powered_down == []


def test_a_hold_is_capped():
    b, _ = _warm_backend()

    b.hold_depth_camera(1e9)

    import time
    assert b._oakd_hold_until - time.monotonic() <= rpi.OAKD_HOLD_MAX_S


def test_a_warmup_with_no_capture_after_it_rearms_the_powerdown():
    # The fleet warms the camera when a session opens. If no take follows, the
    # camera must not be left on forever.
    b, _ = _warm_backend()
    b._oakd = types.SimpleNamespace(is_initialized=True, wait_until_ready=lambda t: True)
    b.raise_if_capture_blocked = lambda: None

    async def scenario():
        await b.prepare_capture()
        armed = b._oakd_keepalive_task is not None
        b._cancel_oakd_keepalive()
        return armed

    assert asyncio.run(scenario()) is True


def test_a_release_takes_effect_without_waiting_out_the_old_lease():
    # The fleet releases with ttl_s=0 at session close. A power-down already
    # sleeping until the old deadline must be re-armed, not left to sleep on.
    b, powered_down = _warm_backend(keepalive_s=0.01)

    async def scenario():
        b.hold_depth_camera(30)
        b._schedule_oakd_keepalive()
        await asyncio.sleep(0.05)
        b.hold_depth_camera(0)
        await asyncio.sleep(0.1)

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))
    assert powered_down == [True]


# --- "warm" means converged, not merely powered ------------------------------

def _camera(ready):
    return types.SimpleNamespace(is_initialized=True, wait_until_ready=lambda t: ready)


def test_an_initialized_camera_is_not_warm_until_it_has_converged():
    b = rpi.RpiBackend()
    b._oakd = _camera(ready=True)
    assert b.is_depth_camera_warm is False

    asyncio.run(b._wait_depth_ready())
    assert b.is_depth_camera_warm is True


def test_a_camera_that_never_converged_is_not_warm():
    b = rpi.RpiBackend()
    b._oakd = _camera(ready=False)

    asyncio.run(b._wait_depth_ready())

    assert b.is_depth_camera_warm is False


def test_a_reinitialized_camera_has_to_converge_again():
    b = rpi.RpiBackend()
    b._oakd = _camera(ready=True)
    asyncio.run(b._wait_depth_ready())

    b._oakd = _camera(ready=True)

    assert b.is_depth_camera_warm is False


# --- two warm-ups at once bring the camera up only once ----------------------

def test_concurrent_enables_initialize_the_camera_once():
    import threading
    import time as _time

    b = rpi.RpiBackend()
    inits = []

    def _init():
        inits.append(threading.get_ident())
        _time.sleep(0.05)
        b._oakd = _camera(ready=True)

    b._init_oakd = _init

    async def scenario():
        await asyncio.gather(b.set_oakd_enabled(True), b.set_oakd_enabled(True))

    asyncio.run(scenario())
    assert len(inits) == 1
