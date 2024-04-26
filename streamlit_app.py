
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import poisson, norm, chi2

# Defining the data
data = np.array([
    (1, 10), (2, 3), (3, 1), (4, 2), (5, 4), (6, 2), (7, 2), (8, 1), (9, 3), (10, 4),
    (11, 4), (12, 2), (13, 1), (14, 3), (15, 1), (16, 4), (17, 3), (18, 4), (19, 3),
    (20, 0), (21, 3), (22, 2), (23, 1), (24, 6), (25, 2), (26, 3), (27, 4), (28, 1),
    (29, 2), (30, 3), (31, 1), (32, 0), (33, 6), (34, 3), (35, 1), (36, 2), (37, 3),
    (38, 4), (39, 3), (40, 2), (41, 1), (42, 2), (43, 0), (44, 3), (45, 2), (46, 1),
    (47, 3), (48, 1), (49, 2), (50, 2), (51, 4), (52, 1), (53, 5), (54, 0), (55, 1),
    (56, 3), (57, 4), (58, 2), (59, 0), (60, 2), (61, 6), (62, 1), (63, 5), (64, 4),
    (65, 3), (66, 8), (67, 3), (68, 0), (69, 0), (70, 0), (71, 2), (72, 2), (73, 5),
    (74, 2), (75, 2), (76, 3), (77, 1), (78, 5), (79, 2), (80, 3), (81, 4), (82, 1),
    (83, 3), (84, 1), (85, 2), (86, 3), (87, 1), (88, 3), (89, 2), (90, 0), (91, 3),
    (92, 1), (93, 2), (94, 2), (95, 3), (96, 3), (97, 3), (98, 4), (99, 3), (100, 2),
    (101, 4), (102, 1), (103, 3), (104, 0), (105, 2), (106, 1), (107, 3), (108, 1),
    (109, 2), (110, 1), (111, 1), (112, 1), (113, 2), (114, 4), (115, 1), (116, 6),
    (117, 1), (118, 7), (119, 2), (120, 0), (121, 3), (122, 4), (123, 3), (124, 7),
    (125, 6), (126, 5), (127, 2), (128, 1), (129, 2), (130, 3), (131, 1), (132, 0),
    (133, 3), (134, 3), (135, 4), (136, 1), (137, 4), (138, 4), (139, 2), (140, 0),
    (141, 0), (142, 1), (143, 5), (144, 3), (145, 2), (146, 5), (147, 2), (148, 4),
    (149, 2), (150, 3), (151, 2), (152, 3), (153, 1), (154, 3), (155, 4), (156, 4),
    (157, 2), (158, 2), (159, 3), (160, 3), (161, 2), (162, 4), (163, 1), (164, 3),
    (165, 2), (166, 0), (167, 4), (168, 3), (169, 1), (170, 0), (171, 0), (172, 1),
    (173, 4), (174, 3), (175, 2), (176, 2), (177, 0), (178, 2), (179, 1), (180, 3),
    (181, 1), (182, 2), (183, 4), (184, 3), (185, 3), (186, 3), (187, 11), (188, 5),
    (189, 3), (190, 2), (191, 1), (192, 4), (193, 6), (194, 25), (195, 3), (196, 1),
    (197, 6), (198, 1), (199, 1), (200, 1), (201, 4), (202, 1), (203, 1), (204, 1),
    (205, 1), (206, 8), (207, 4), (208, 2), (209, 2), (210, 1), (211, 3), (212, 2),
    (213, 0), (214, 3), (215, 1), (216, 1), (217, 2), (218, 5), (219, 3), (220, 5),
    (221, 4), (222, 1), (223, 4), (224, 2), (225, 1), (226, 2), (227, 4), (228, 1),
    (229, 3), (230, 3), (231, 2), (232, 3), (233, 4), (234, 1), (235, 1), (236, 2),
    (237, 4), (238, 2), (239, 4), (240, 4), (241, 3), (242, 3), (243, 2), (244, 1),
    (245, 0), (246, 2), (247, 2), (248, 3), (249, 3), (250, 1)
])

# Extracting counts and bin positions
positions, counts = data[:, 0], data[:, 1]

# Fitting Poisson distribution
poisson_fit_lambda = np.mean(counts)
poisson_fit = poisson.pmf(positions, poisson_fit_lambda)

# Fitting Gaussian (Normal) distribution
normal_fit_mean, normal_fit_std = np.mean(counts), np.std(counts)
normal_fit = norm.pdf(positions, normal_fit_mean, normal_fit_std)

# Computing chi-square for Poisson fit
observed_counts = counts
expected_counts_poisson = poisson_fit_lambda * len(data)
chi_square_poisson, p_value_poisson = chi2.chisquare(observed_counts, expected_counts_poisson)

# Computing chi-square for Gaussian fit
expected_counts_normal = normal_fit * len(data)
chi_square_normal, p_value_normal = chi2.chisquare(observed_counts, expected_counts_normal)

# Displaying results in Streamlit
st.write(f"Mean of the data: {normal_fit_mean}")
st.write(f"Standard deviation of the data: {normal_fit_std}")
st.write(f"Chi-square value for Poisson fit: {chi_square_poisson}")
st.write(f"P-value for Poisson fit: {p_value_poisson}")
st.write(f"Chi-square value for Gaussian fit: {chi_square_normal}")
st.write(f"P-value for Gaussian fit: {p_value_normal}")

# Plotting the data and fits
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(positions, counts, label='Data')
plt.plot(positions, poisson_fit * len(data), 'r-', label='Poisson Fit')
plt.plot(positions, normal_fit * len(data), 'g--', label='Gaussian Fit')
plt.xlabel('Decay')
plt.ylabel('Frequency')
plt.title('Data Distribution with Poisson and Gaussian Fits')
plt.legend()
st.pyplot(plt)


# Ejecutar la aplicación
if __name__ == '__main__':
    main()
