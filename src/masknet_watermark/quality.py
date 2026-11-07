"""Paired audio measurements for this watermark's evaluation records."""
import math
import statistics
import warnings

def measure(reference, marked, sample_rate):
    import numpy as np
    from pesq import pesq
    from pystoi import stoi
    ref, deg = np.asarray(reference, np.float64), np.asarray(marked, np.float64)
    if ref.ndim != 1 or deg.shape != ref.shape or sample_rate != 16000:
        raise ValueError("Quality inputs must be aligned mono audio at 16 kHz")
    signal = float(np.mean(ref**2))
    noise = float(np.mean((ref-deg)**2))
    result = {"snr_db": 10*math.log10(signal/noise) if signal > 0 and noise > 0 else None,
              "exact_copy": noise == 0, "pesq_wb": None, "stoi": None, "errors": {}}
    for name, function in [("pesq_wb", lambda: pesq(16000, ref, deg, "wb")),
                           ("stoi", lambda: stoi(ref, deg, 16000, extended=False))]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = float(function())
            if any("Not enough STFT frames" in str(w.message) for w in caught):
                raise ValueError("Insufficient voiced frames")
            if not math.isfinite(value):
                raise ValueError("Nonfinite quality value")
            result[name] = value
        except Exception as error:
            result["errors"][name] = type(error).__name__ + ": " + str(error)
    return result

def summarize(records):
    summaries = {}
    for condition in dict.fromkeys(v["condition"] for v in records):
        rows = [v for v in records if v["condition"] == condition]
        entry = {"source_N": len(rows), "speaker_N": len({v["speaker_id"] for v in rows}),
                 "peak_max": max(v["peak"] for v in rows),
                 "fraction_outside_unit_mean": statistics.mean(v["fraction_outside_unit"] for v in rows)}
        for block in ("incremental", "total"):
            values = [v[block] for v in rows if v[block] is not None]
            entry[block] = {"paired_N": len(values), "missing_reference_N": len(rows)-len(values)}
            for metric in ("snr_db", "pesq_wb", "stoi"):
                valid = [v[metric] for v in values if v[metric] is not None]
                entry[block][metric] = {"valid_N": len(valid), "missing_N": len(values)-len(valid),
                                        "mean": statistics.mean(valid) if valid else None,
                                        "median": statistics.median(valid) if valid else None,
                                        "min": min(valid) if valid else None,
                                        "max": max(valid) if valid else None}
        summaries[condition] = entry
    return summaries
