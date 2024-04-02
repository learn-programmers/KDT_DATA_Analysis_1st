import matplotlib.pyplot as plt
import seaborn as sns

def generate_colors(num_colors):
    # tab10에서는 10가지의 고유한 색상을 제공하므로 이를 활용하여 색상을 생성합니다.
    colors = sns.color_palette("tab10", num_colors)
    return colors

def showplot(columnname, train):
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax = ax.flatten()
    value_counts = train[columnname].value_counts()
    labels = value_counts.index.tolist()

    # 색상 생성
    colors = generate_colors(len(value_counts))
    color_map = dict(zip(labels, colors))

    # Donut Chart
    wedges, texts, autotexts = ax[0].pie(
        value_counts, autopct='%1.1f%%', textprops={'size': 9, 'color': 'white', 'fontweight': 'bold'},
        colors=[color_map[x] for x in labels], wedgeprops=dict(width=0.35), startangle=80, pctdistance=0.85)

    # Circle
    centre_circle = plt.Circle((0, 0), 0.6, fc='white')
    ax[0].add_artist(centre_circle)

    # Count Plot
    colormap = {label: color for label, color in zip(labels, colors)}
    sns.countplot(data=train, y=columnname, ax=ax[1], order=labels, hue=columnname, palette=colormap, legend=False)
    for i, v in enumerate(value_counts):
        ax[1].text(v + 1, i, str(v), color='black', fontsize=10, va='center')
    sns.despine(left=True, bottom=True)
    plt.yticks(fontsize=9, color='black')
    ax[1].set_ylabel(None)
    plt.xlabel("")
    plt.xticks([])
    fig.suptitle(columnname, fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()