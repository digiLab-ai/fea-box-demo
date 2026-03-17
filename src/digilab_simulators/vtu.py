from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


def write_vtu(output: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    points = output["points"]
    cells = output["cells"]
    cell_types = output["cell_types"]
    point_temperature = output["point_data"]["temperature"]
    cell_temperature = output["cell_data"]["temperature"]

    connectivity = " ".join(" ".join(str(i) for i in cell) for cell in cells)
    offsets_list = []
    offset = 0
    for cell in cells:
        offset += len(cell)
        offsets_list.append(str(offset))
    offsets = " ".join(offsets_list)
    types = " ".join(str(t) for t in cell_types)
    points_text = " ".join(f"{p[0]} {p[1]} {p[2]}" for p in points)
    point_temp_text = " ".join(str(v) for v in point_temperature)
    cell_temp_text = " ".join(str(v) for v in cell_temperature)

    xml = f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)}">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {escape(points_text)}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          {escape(connectivity)}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          {escape(offsets)}
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          {escape(types)}
        </DataArray>
      </Cells>
      <PointData Scalars="temperature">
        <DataArray type="Float64" Name="temperature" format="ascii">
          {escape(point_temp_text)}
        </DataArray>
      </PointData>
      <CellData Scalars="temperature">
        <DataArray type="Float64" Name="temperature" format="ascii">
          {escape(cell_temp_text)}
        </DataArray>
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
'''
    out_path.write_text(xml, encoding="utf-8")
    return out_path
