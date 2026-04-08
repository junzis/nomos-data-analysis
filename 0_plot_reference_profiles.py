"""Generate the NADP reference speed profiles diagram (plot 00)."""

import matplotlib.pyplot as plt
import numpy as np

alt_grid = np.arange(200, 3600, 100)


def interp(alts, vals):
    return np.interp(alt_grid, alts, vals)


# NADP1
nadp1 = interp([0, 800, 3000, 3500], [0, 0, 0, 25])

# NADP2 sub-types
nadp2_800 = interp([0, 800, 3000, 3500], [0, 0, 70, 75])
nadp2_1000 = interp([0, 1000, 3000, 3500], [0, 0, 70, 75])
nadp2_1500 = interp([0, 1500, 3000, 3500], [0, 0, 70, 75])

# Single-panel: 20cm wide (renders to 10cm in docx at 2x)
FIG_W1 = 20 / 2.54  # inches
fig, ax = plt.subplots(figsize=(FIG_W1, FIG_W1 * 0.75))

# NADP1
ax.plot(nadp1, alt_grid, color="#4C72B0", lw=3, label="NADP1 (close-in)")

# NADP2 sub-types
ax.plot(nadp2_800, alt_grid, color="#DD5145", lw=2.5, label="NADP2-800")
ax.plot(nadp2_1000, alt_grid, color="#FF7F0E", lw=2.5, label="NADP2-1000")
ax.plot(nadp2_1500, alt_grid, color="#8C564B", lw=2.5, label="NADP2-1500")

# Annotation lines (colored to match sub-type where applicable)
hline_colors = {800: "#DD5145", 1000: "#FF7F0E", 1500: "#8C564B", 3000: "gray"}
for alt, text in [(800, "800 ft"), (1000, "1000 ft"), (1500, "1500 ft"), (3000, "3000 ft")]:
    c = hline_colors[alt]
    ax.axhline(alt, color=c, ls=":", lw=0.8, alpha=0.5)
    ax.text(82, alt + 30, text, fontsize=9, color=c, alpha=0.7)

ax.set_xlabel("IAS - V2 (kt)")
ax.set_ylabel("Altitude above aerodrome (ft)")
ax.set_title("NADP Reference Speed Profiles")
ax.legend(loc="upper left")
ax.set_xlim(-5, 90)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/00_reference_profiles.png", dpi=150)
print("Saved plots/00_reference_profiles.png")
