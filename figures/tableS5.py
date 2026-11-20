"""Table S5: board designs (RFSoC 4x2 board, XCZU48DR chip with 425,280 LUTs)."""
import _style as S


def main():
    t = S.read("tables", "tableS5.csv")
    rows = []
    for r in t.itertuples():
        unit = "µs"
        time = f"{r.time_us:,.2f} {unit}" if r.mode == "one_shot" else f"{r.time_us:,.1f} {unit}"
        power = (">= " if r.power_is_lower_bound == "yes" else "") + f"{r.power_decoding_w:.1f} / {r.power_idle_w:.1f}"
        rows.append(["One syndrome per shot" if r.mode == "one_shot" else "Real time", r.d, r.replicas, r.layout,
                     f"{r.luts:,} ({r.lut_percent:.1f})", f"{r.clock_mhz:.1f}", time, power])
    S.write_table("tableS5", ["Mode", "d", "Replicas", "Layout", "LUTs (%)", "Clock (MHz)", "Time", "Power (W)"], rows,
                  note="Time: per shot at the fixed budget (one syndrome per shot) or per round (real time, 2^12 sweeps "
                       "per window). Power: core rail, decoding / idle.")


if __name__ == "__main__":
    main()
