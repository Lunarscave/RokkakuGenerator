from strokes_generator import StrokesGenerator

OUTPUT_PATH = "D:\\ResearchProjects\\BuildingSketch\\Datasets\\TrainHDF5\\hip"
STROKES_NUMS = 2
strokes_generator = StrokesGenerator()
for i in range(STROKES_NUMS):
    strokes_generator.get_geometry_strokes("hip", f"{OUTPUT_PATH}\\{i + 1}.hdf5")


