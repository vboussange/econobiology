#=
Settings for pyplot
=#

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
mplt = pyimport("mpl_toolkits.axes_grid1.inset_locator")

rcParams["font.size"] = 9
rcParams["axes.titlesize"] = 10
rcParams["axes.labelsize"] = 10
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["legend.fontsize"] = 8
rcParams["figure.titlesize"] = 10
rcParams["lines.markersize"] = 6

MARKERSTYLESMODELS = ["o", "v", "s", "*", "h"]
COLORMODELS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
LINESTYLES = ["-", "--", ":", "-.", (0, (5,10))]
