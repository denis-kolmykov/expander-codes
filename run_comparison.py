"""Пример сравнения экспандерных кодов и правил декодирования в BSC."""

from expander_codes import (
    BSC,
    ExpanderCode,
    noise_threshold,
    print_table,
    save_rows,
    sweep,
)

SEED = 2026
LEVELS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]


def build_codes():
    codes = [
        ExpanderCode.random_regular(n=1008, dv=3, dc=6, seed=SEED + 1, flip_mode="max"),
        ExpanderCode.random_regular(n=1008, dv=4, dc=8, seed=SEED + 2, flip_mode="max"),
        ExpanderCode.random_regular(n=1008, dv=3, dc=4, seed=SEED + 3, flip_mode="max"),
    ]
    base = codes[0]
    for mode in ("serial", "majority"):
        codes.append(ExpanderCode(
            base.H,
            name=f"expander_n1008_dv3_dc6_{mode}",
            max_iter=base.max_iter,
            flip_mode=mode,
            meta=base.meta,
        ))
    return codes


def main():
    codes = build_codes()

    rows = sweep(
        codes,
        BSC,
        LEVELS,
        seed=SEED,
        batch_size=500,
        max_blocks=20_000,
        target_block_errors=100,
    )

    print("характеристики графов:")
    for code in codes:
        print(f"  {code.name}: обхват {code.girth}, расширение {code.expansion:.3f}")
    print()

    print_table(rows)
    path = save_rows(rows, "results.csv")
    print(f"\nрезультаты записаны в {path}")

    print("\nпорог шума при FER <= 1e-3:")
    for code in codes:
        print(f"  {code.name}: {noise_threshold(rows, code.name, 1e-3)}")


if __name__ == "__main__":
    main()
