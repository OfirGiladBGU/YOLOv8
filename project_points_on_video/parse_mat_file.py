# OPTION 1

# import scipy.io
# import pandas as pd
#
#
# if __name__ == '__main__':
#     mat_filepath = r"C:\Users\Ofir Gilad\PycharmProjects\YOLOv8\project_points_on_video\input\control_dinstein_girls.mat"
#     mat = scipy.io.loadmat(mat_filepath)
#
#     df = pd.DataFrame(mat)
#     csv_filepath = r"C:\Users\Ofir Gilad\PycharmProjects\YOLOv8\project_points_on_video\output\control_dinstein_girls.csv"
#     df.to_csv('your_data.csv')  # your_data.csv final data in csv file

# OPTION 2

import scipy.io
import pandas as pd


if __name__ == '__main__':
	mat_filepath = r"C:\Users\Ofir Gilad\PycharmProjects\YOLOv8\project_points_on_video\control_dinstein_girls.mat"
	mat = scipy.io.loadmat(mat_filepath)

	with open("./output/out.txt", "w") as f:
		f.write(str(mat))
