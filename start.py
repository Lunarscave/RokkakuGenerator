from strokes_generator import StrokesGenerator

OUTPUT_PATH = "D:\\ResearchProjects\\BuildingSketch\\Datasets\\TrainHDF5\\freeform"
STROKES_NUMS = 3700
strokes_generator = StrokesGenerator()
# strokes_generator.get_all_geometries_strokes(f"{OUTPUT_PATH}", STROKES_NUMS)
for i in range(1, STROKES_NUMS):
    print(i)
    strokes_generator.get_geometry_strokes("freeform", f"{OUTPUT_PATH}\\{i + 1}.hdf5")

