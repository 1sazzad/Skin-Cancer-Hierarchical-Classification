from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/phase05_final_scientific_synthesis/figures")
OUT.mkdir(parents=True, exist_ok=True)

systems = ["Flat\n4-class", "Hierarchy\npredicted gate", "Hierarchy\noracle gate"]
macro_f1 = [0.6192224685168973, 0.5685909456725847, 0.7769758578867025]

plt.figure(figsize=(7.5, 4.8))
x = np.arange(len(systems))
bars = plt.bar(x, macro_f1)
plt.xticks(x, systems)
plt.ylabel("Macro-F1")
plt.ylim(0, 0.85)
plt.title("End-to-End Four-Class Performance")
for bar, value in zip(bars, macro_f1):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
plt.tight_layout()
plt.savefig(OUT / "fig1_end_to_end_macro_f1.png", dpi=300, bbox_inches="tight")
plt.close()

predicted = 0.5685909456725847
oracle = 0.7769758578867025
plt.figure(figsize=(6.6, 4.8))
bars = plt.bar(["Predicted gate", "Oracle gate"], [predicted, oracle])
plt.ylabel("Macro-F1")
plt.ylim(0, 0.85)
plt.title("Routing Error Decomposition")
for bar, value in zip(bars, [predicted, oracle]):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
plt.text(0.5, (predicted + oracle) / 2, f"Routing loss = {oracle - predicted:.3f}", ha="center")
plt.tight_layout()
plt.savefig(OUT / "fig2_routing_loss.png", dpi=300, bbox_inches="tight")
plt.close()

classes = ["Non-malignant", "Melanoma", "BCC", "SCC"]
flat_f1 = [0.8218298555377207, 0.6094215861657722, 0.6884955752212389, 0.35714285714285715]
hierarchy_f1 = [0.7785969084423305, 0.5456688273423689, 0.6591889559965487, 0.2909090909090909]

plt.figure(figsize=(8.2, 5.0))
x = np.arange(len(classes))
w = 0.36
plt.bar(x - w / 2, flat_f1, width=w, label="Flat")
plt.bar(x + w / 2, hierarchy_f1, width=w, label="Hierarchy")
plt.xticks(x, classes)
plt.ylabel("F1")
plt.ylim(0, 0.9)
plt.title("Per-Class F1: Flat vs Deployed Hierarchy")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig3_classwise_f1.png", dpi=300, bbox_inches="tight")
plt.close()

delta = np.array([-0.043232947095390162, -0.063752758823403366, -0.029306619224690245, -0.066233766233766256])
ci_low = np.array([-0.056034654679731823, -0.088292991318156988, -0.05623762052919197, -0.15000000000000002])
ci_high = np.array([-0.030510378896148188, -0.039764985320049108, -0.0027217029631088222, 0.0195857495883342])

plt.figure(figsize=(7.6, 4.8))
y = np.arange(len(classes))
err_low = delta - ci_low
err_high = ci_high - delta
plt.errorbar(delta, y, xerr=np.vstack([err_low, err_high]), fmt="o", capsize=4)
plt.axvline(0, linewidth=1)
plt.yticks(y, classes)
plt.xlabel("F1 difference (Hierarchy − Flat)")
plt.title("Paired Classwise F1 Differences with 95% CIs")
plt.tight_layout()
plt.savefig(OUT / "fig4_classwise_delta_ci.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved publication figures to {OUT}")
