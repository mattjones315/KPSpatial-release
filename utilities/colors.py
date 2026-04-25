import seaborn as sns

DISCRETE_COLORS = {
1:["black"],
2:["#CD2626", "#1874CD"],
3:["#CD2626", "#FFE600", "#1874CD"],
4:["#CD2626", "#FFE600", "#009E73", "#1874CD"],
5:["#CD2626", "#FFE600", "#009E73", "#1874CD", "#8E0496"],
6:["#CD2626", "#E69F00", "#FFE600", "#009E73", "#1874CD", "#8E0496"],
7:["#CD2626", "#E69F00", "#FFE600", "#009E73", "#83A4FF", "#1874CD", "#8E0496"],
8:["#CD2626", "#E69F00", "#FFE600", "#009E73", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
9:["#CD2626", "#E69F00", "#FFE600", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
10:["#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
11:["#FF7D7D", "#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
12:["#FF7D7D", "#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
13:["#FF7D7D", "#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
14:["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
15:["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
16:["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2"],
17:["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF"],
18:["#FF7D7D", "#CD2626", "#7F0303", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF"],
19:["#FF7D7D", "#CD2626", "#7F0303", "#D34818", "#E69F00", "#845C44", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF"],
20:["#FF7D7D", "#CD2626", "#7F0303", "#D34818", "#E69F00", "#845C44", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF", "#343434"]}

DISCRETE_CMAP = {k:sns.color_palette(v) for k,v in DISCRETE_COLORS.items()}

MODULE_COLORS = {1: "#CE2628", 2: "#F37B7D", 3: "#e5a024", 4: "#89C75F", 5: "#009E73",
                6: "#8FD6E5", 7: "#6E4B9E", 8: "#899FD2", 9: "#89288f", 10: "#3173ba",
                11: "#c16cac"}

SLIDETAGS_COLORS = {'AT1-like': '#d60000',
                      'AT2': "#907089",
                      'AT2-like': '#ac9c00',
                      'Ciliated cell': "#740070",
                      'Club cell': "#cf0080",
                      'Early EMT': "#ff9a89",
                      'Early gastric': "#E1BD6D",
                      'Endoderm-like': "#ff4d6b",
                      'Gastric-like': "#E58601",
                      'High-plasticity cell state': "#ba6b2f",
                      'Late gastric': '#c6bf91',
                      'Neuronal-like': "#d48ab3",
                      'Pre-EMT': "#e4bf00",
                      'Alveolar Macrophage': "#5BBCD6",
                      'B cell': "#00A08A",
                      'Endothelial': "#ABDDDE",
                      'Fibroblast': "#005760",
                      'Mac': "#0179aa",
                      'Mesothelial': "#bcdbff",
                      'MonoMac': "#41a3ff",
                      'Muscle': "#074f95",
                      'Pecam1+ Mac': '#91a5ba',
                      'Pericyte': "#9aba9e",
                      'migDC': "#aca3ff"}

TUMOR_ZSTACK_COLORS = {}
