"""Use Windows' system ICU when Python has no installed IANA database."""

import ctypes
from datetime import datetime, timedelta
from zoneinfo import ZoneInfoNotFoundError


def local_time(instant: datetime, zone: str) -> datetime:
    """Convert a UTC instant to the wall clock in a named IANA zone."""
    # LOAD_LIBRARY_SEARCH_SYSTEM32 restricts lookup to the system directory.
    icu = ctypes.CDLL("icu.dll", winmode=0x800)
    status = ctypes.c_int32()
    status_ptr = ctypes.POINTER(ctypes.c_int32)
    wchar = ctypes.c_wchar_p
    system_id = ctypes.c_int8()
    canonical = ctypes.create_unicode_buffer(256)
    icu.ucal_getCanonicalTimeZoneID.argtypes = [
        wchar,
        ctypes.c_int32,
        wchar,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int8),
        status_ptr,
    ]
    icu.ucal_getCanonicalTimeZoneID(
        zone, -1, canonical, len(canonical), ctypes.byref(system_id), ctypes.byref(status)
    )
    if status.value > 0 or not system_id.value or "\0" in zone:
        raise ZoneInfoNotFoundError(f"Unknown time zone: {zone!r}")

    icu.ucal_open.argtypes = [wchar, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int, status_ptr]
    icu.ucal_open.restype = ctypes.c_void_p
    icu.ucal_setMillis.argtypes = [ctypes.c_void_p, ctypes.c_double, status_ptr]
    icu.ucal_setMillis.restype = None
    icu.ucal_get.argtypes = [ctypes.c_void_p, ctypes.c_int, status_ptr]
    icu.ucal_close.argtypes = [ctypes.c_void_p]
    icu.ucal_close.restype = None
    calendar = icu.ucal_open(canonical, -1, b"en", 1, ctypes.byref(status))
    if not calendar:
        raise ValueError(f"ICU could not open the calendar for {zone!r}")
    try:
        icu.ucal_setMillis(calendar, instant.timestamp() * 1000, ctypes.byref(status))
        # UCalendarDateFields UCAL_ZONE_OFFSET and UCAL_DST_OFFSET (milliseconds).
        standard = icu.ucal_get(calendar, 15, ctypes.byref(status))
        daylight = icu.ucal_get(calendar, 16, ctypes.byref(status))
        if status.value > 0:
            raise ValueError(f"ICU could not resolve the time in {zone!r}")
        return instant.replace(tzinfo=None) + timedelta(milliseconds=standard + daylight)
    finally:
        icu.ucal_close(calendar)
