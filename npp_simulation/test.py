#!/usr/bin/env python

# test_fixed.py
# Minimal working OpenMC example: 3x3 assembly, central control rod
# Partial insertion: CR present only in upper axial half.
import openmc

# -----------------------
# 1) Materials
# -----------------------
fuel = openmc.Material(name='UO2 fuel')
fuel.add_element('U', 1.0, enrichment=4.0)
fuel.add_element('O', 2.0)
fuel.set_density('g/cm3', 10.4)

clad = openmc.Material(name='Zry-4')
clad.add_element('Zr', 1.0)
clad.set_density('g/cm3', 6.55)

water = openmc.Material(name='H2O')
water.add_element('H', 2.0)
water.add_element('O', 1.0)
water.set_density('g/cm3', 0.745)
water.add_s_alpha_beta('c_H_in_H2O')

cr_mat = openmc.Material(name='B4C absorber')
cr_mat.add_element('B', 4.0)
cr_mat.add_element('C', 1.0)
cr_mat.set_density('g/cm3', 2.52)

mats = openmc.Materials([fuel, clad, water, cr_mat])
mats.export_to_xml()

# -----------------------
# 2) Geometry primitives (radii, planes)
# -----------------------
r_fuel = 0.41      # fuel pellet radius [cm]
r_clad = 0.47      # cladding outer radius [cm]
cr_radius = 0.5    # control rod radius [cm]
pitch = 1.26       # lattice pitch [cm]

# Axial planes (z coordinates) for partial insertion
z_bottom = 0.0
z_mid = 50.0
z_top = 100.0

z0 = openmc.ZPlane(z0=z_bottom, boundary_type='reflective')
z_mid_plane = openmc.ZPlane(z0=z_mid)
z1 = openmc.ZPlane(z0=z_top, boundary_type='reflective')

# Bounding planes for the assembly (square region)
half_extent = 1.5 * pitch
x_min = openmc.XPlane(x0=-half_extent, boundary_type='reflective')
x_max = openmc.XPlane(x0= half_extent, boundary_type='reflective')
y_min = openmc.YPlane(y0=-half_extent, boundary_type='reflective')
y_max = openmc.YPlane(y0= half_extent, boundary_type='reflective')

# Cylinders for pins
fuel_or = openmc.ZCylinder(r=r_fuel)
clad_or = openmc.ZCylinder(r=r_clad)
cr_or = openmc.ZCylinder(r=cr_radius)

# -----------------------
# 3) Cells for a fuel pin
# -----------------------
fuel_cell = openmc.Cell(name='fuel')
fuel_cell.fill = fuel
fuel_cell.region = -fuel_or

clad_cell = openmc.Cell(name='clad')
clad_cell.fill = clad
clad_cell.region = +fuel_or & -clad_or

moderator_cell = openmc.Cell(name='moderator')
moderator_cell.fill = water
moderator_cell.region = +clad_or  # will be limited by lattice bounding cell

# -----------------------
# 4) Cells for a control-rod pin
# -----------------------
cr_cell = openmc.Cell(name='cr_absorber')
cr_cell.fill = cr_mat
cr_cell.region = -cr_or

cr_moderator = openmc.Cell(name='cr_mod')
cr_moderator.fill = water
cr_moderator.region = +cr_or

# -----------------------
# 5) Pin universes
# -----------------------
fuel_univ = openmc.Universe(name='fuel_pin')
fuel_univ.add_cells([fuel_cell, clad_cell, moderator_cell])

cr_univ = openmc.Universe(name='cr_pin')
cr_univ.add_cells([cr_cell, cr_moderator])

# -----------------------
# 6) Lattices (3x3) with lower_left and shape set
# -----------------------
# Top-level lattice used in upper region (center is CR)
lat_upper = openmc.RectLattice(name='lat_upper')
lat_upper.pitch = (pitch, pitch)
lat_upper.lower_left = (x_min.x0, y_min.y0)   # numeric coords of lower-left corner
# lat_upper.shape = (3, 3)
lat_upper.universes = [
    [fuel_univ, fuel_univ, fuel_univ],
    [fuel_univ, cr_univ,   fuel_univ],
    [fuel_univ, fuel_univ, fuel_univ]
]

# Lattice for lower region (center is fuel, CR absent)
lat_lower = openmc.RectLattice(name='lat_lower')
lat_lower.pitch = (pitch, pitch)
lat_lower.lower_left = (x_min.x0, y_min.y0)
# lat_lower.shape = (3, 3)
lat_lower.universes = [
    [fuel_univ, fuel_univ, fuel_univ],
    [fuel_univ, fuel_univ, fuel_univ],
    [fuel_univ, fuel_univ, fuel_univ]
]

# Cells that contain the lattice and are bounded laterally by the planes
lat_upper_cell = openmc.Cell(name='lat_upper_cell')
lat_upper_cell.region = +x_min & -x_max & +y_min & -y_max
lat_upper_cell.fill = lat_upper

lat_lower_cell = openmc.Cell(name='lat_lower_cell')
lat_lower_cell.region = +x_min & -x_max & +y_min & -y_max
lat_lower_cell.fill = lat_lower

# -----------------------
# 7) Root axial cells (partial insertion)
# -----------------------
root_lower = openmc.Cell(name='root_lower')
root_lower.region = +z0 & -z_mid_plane
root_lower.fill = openmc.Universe(name='lower_universe')
root_lower.fill.add_cell(lat_lower_cell)

root_upper = openmc.Cell(name='root_upper')
root_upper.region = +z_mid_plane & -z1
root_upper.fill = openmc.Universe(name='upper_universe')
root_upper.fill.add_cell(lat_upper_cell)

# Build the root universe and geometry
root_univ = openmc.Universe(name='root')
root_univ.add_cells([root_lower, root_upper])

geom = openmc.Geometry(root_univ)
geom.export_to_xml()

# -----------------------
# 8) Settings and a simple tally
# -----------------------
settings = openmc.Settings()
settings.batches = 40
settings.inactive = 10
settings.particles = 2000
settings.source = openmc.Source(space=openmc.stats.Point((0.0, 0.0, (z_top+z_bottom)/2.0)))
settings.export_to_xml()

# Tally: flux in the entire lattice box (uses the lat_upper_cell spatial cell)
t = openmc.Tally(name='assembly_flux')
t.filters = [openmc.CellFilter([lat_upper_cell])]
t.scores = ['flux']
tallies = openmc.Tallies([t])
tallies.export_to_xml()

print("Exported materials.xml, geometry.xml, settings.xml, tallies.xml")
print("Run `openmc` to execute the simulation.")
openmc.run()

