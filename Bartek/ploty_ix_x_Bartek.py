import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation

# zaladuj dane
file = r"C:\Users\Bartosz\Downloads\exciton_on_atoms.dat"

pierw, x, y, z, psi2 = [], [], [], [], []

with open(file) as f:
    for line in f:
        p = line.split()
        pierw.append(p[0])
        x.append(float(p[1]))
        y.append(float(p[2]))
        z.append(float(p[3]))
        psi2.append(float(p[4]))

pierw = np.array(pierw)
x = np.array(x)
y = np.array(y)
z = np.array(z)
psi2 = np.array(psi2)

# WS2 i WSe2 osobne ploty

mask_ws2 = np.isin(pierw, ["W"])
mask_wse2 = np.isin(pierw, ["Se"])

plt.figure()
plt.scatter(x[mask_ws2], y[mask_ws2], c=psi2[mask_ws2], cmap="viridis", s=25)
plt.colorbar(label="|psi|2")
plt.title("WS2 layer exciton density")
plt.axis("equal")
plt.show()

plt.figure()
plt.scatter(x[mask_wse2], y[mask_wse2], c=psi2[mask_wse2], cmap="viridis", s=25)
plt.colorbar(label="|psi|2")
plt.title("WSe2 layer exciton density")
plt.axis("equal")
plt.show()


# wersja wykresu ciaglego

grid_x = np.linspace(x.min(), x.max(), 250)
grid_y = np.linspace(y.min(), y.max(), 250)

Xg, Yg = np.meshgrid(grid_x, grid_y)

psi_grid = griddata(
    (x, y),
    psi2,
    (Xg, Yg),
    method="cubic"
)

plt.figure()
plt.contourf(Xg, Yg, psi_grid, levels=120, cmap="viridis")
plt.colorbar(label="|psi|2")
plt.axis("equal")
plt.title("2D continuous exciton density (interpolation)")
plt.show()

# eksportowanie do VESTA - NIE wiem czy bedzie jak powinno

out_xyz = r"C:\Users\Bartosz\Music\hee_hee.txt"

with open(out_xyz, "w") as f:
    f.write(f" row count: {len(x)}\n")
    f.write("Exciton density projection\n")
    for t, xi, yi, zi, v in zip(pierw, x, y, z, psi2):
        f.write(f"{t} {xi:.6f} {yi:.6f} {zi:.6f} {v:.8e}\n")

print("Saved XYZ:", out_xyz)

# wykres 3d z warstw

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(x, y, z, c=psi2, cmap="viridis", s=20)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.colorbar(sc)
plt.title("3D exciton density on structure")
plt.show()