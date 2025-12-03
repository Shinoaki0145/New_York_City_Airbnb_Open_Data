import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['lines.solid_capstyle'] = 'butt' 

def plot_missing_values(data, column_names, target_columns, title='Missing Values'):
    """
    Plot missing values chart (Fixed display: Auto-scale based on actual data)
    """
    missing_counts = []
    cols_with_missing = []
    
    for col_name in target_columns:
        if col_name in column_names:
            col_idx = np.where(column_names == col_name)[0][0]
            col_data = data[:, col_idx]
            missing = np.sum(col_data == '')
            missing_counts.append(missing)
            cols_with_missing.append(col_name)
    
    if len(missing_counts) == 0:
        print(f"No missing values in columns: {target_columns}")
        return

    missing_counts = np.array(missing_counts)
    cols_with_missing = np.array(cols_with_missing)
    total_rows = data.shape[0]
    
    max_missing = np.max(missing_counts)
    
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    sns.barplot(
        x=missing_counts, 
        y=cols_with_missing, 
        hue=cols_with_missing, 
        legend=False, 
        palette='Reds_r', 
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Number of missing values')

    if max_missing > 0:
        ax.set_xlim(0, max_missing * 1.15)
    
    for p in ax.patches:
        width = p.get_width()
        if width > 0: 
            percent = (width / total_rows * 100)
            ax.text(width + (max_missing * 0.01), p.get_y() + p.get_height()/2, 
                    f'{int(width):,} ({percent:.3f}%)', 
                    va='center', fontsize=10, fontweight='bold', color='black')
    
    plt.show()


def plot_distribution(data, column_name, bins=50, title=None):
    """
    Plot distribution and boxplot (Fully synchronized with Seaborn)
    """
    data_clean = data[~np.isnan(data)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    sns.histplot(
        x=data_clean, 
        ax=axes[0], 
        stat='density',        
        kde=True, 
        bins=bins,
        edgecolor='white',
        linewidth=0.5
    )

    mean_val = np.mean(data_clean)
    median_val = np.median(data_clean)
    axes[0].axvline(mean_val, color='crimson', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
    axes[0].axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.3f}')
    
    axes[0].set_title(title if title else f"{column_name} Distribution")
    axes[0].legend()
    
    bp = axes[1].boxplot(data_clean, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', edgecolor='blue', linewidth=1.5),
                        medianprops=dict(color='orange', linewidth=2.5),
                        whiskerprops=dict(color='blue', linewidth=1.5),
                        capprops=dict(color='blue', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, 
                                       markeredgecolor='red', alpha=0.6))
    
    axes[1].set_title(f'{column_name} Outlier Detection')
    axes[1].set_ylabel(column_name)
    
    Q1 = np.percentile(data_clean, 25)
    Q3 = np.percentile(data_clean, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    n_outliers = len(data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)])

    axes[1].text(0.95, 0.95, f'Outliers: {n_outliers}', 
                transform=axes[1].transAxes,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(data, column_name, top_n=20):
    """
    Plot categorical variable distribution (Seaborn style)
    """
    unique, counts = np.unique(data, return_counts=True)
    
    sorted_indices = np.argsort(counts)[::-1][:top_n]
    top_categories = unique[sorted_indices]
    top_counts = counts[sorted_indices]

    plt.figure(figsize=(10, 6))

    sns.barplot(x=top_categories, y=top_counts, hue=top_categories, palette="viridis", legend=False)
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column_name} (Top {top_n}) - Bar Chart')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(9, 9))

    colors = sns.color_palette('pastel')[0:len(top_categories)]
    
    wedges, texts, autotexts = plt.pie(
        top_counts, 
        labels=top_categories, 
        autopct='%1.1f%%',
        startangle=0,           
        pctdistance=0.85,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    
    plt.setp(autotexts, size=9, weight="bold", color="black")
    plt.title(f'Percentage of {column_name} (Top {top_n}) - Pie Chart', pad=20)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(data, column_names, figsize=(12, 10)):
    """
    Plot correlation heatmap (Already using Seaborn, keeping but adjusting style)
    """
    correlation_matrix = np.corrcoef(data, rowvar=False)
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                xticklabels=column_names, yticklabels=column_names)
    plt.title('Correlation Matrix', pad=20)
    plt.tight_layout()
    plt.show()


def plot_price_by_category(categories, prices, category_name, top_n=10):
    """
    Plot average price chart (Using Seaborn Barplot)
    """
    unique_cats = np.unique(categories)
    avg_prices = []
    
    for cat in unique_cats:
        mask = categories == cat
        avg_price = np.mean(prices[mask])
        avg_prices.append(avg_price)
    
    avg_prices = np.array(avg_prices)
    
    sorted_indices = np.argsort(avg_prices)[::-1][:top_n]
    top_cats = unique_cats[sorted_indices]
    top_prices = avg_prices[sorted_indices]
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(x=top_prices, y=top_cats, hue=top_cats, palette="Blues_r", legend=False)
    
    ax.set_xlabel('Average Price ($)')
    ax.set_ylabel(category_name)
    ax.set_title(f'Average Price by {category_name} (Top {top_n})')

    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 1, p.get_y() + p.get_height()/2,
                f'${top_prices[i]:.3f}',
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_price_density_by_neighbourhood(neighbourhood_groups, prices):
    """
    Plot violin plot of price distribution by neighbourhood group
    
    Parameters:
    -----------
    neighbourhood_groups : numpy.ndarray
        Array of neighbourhood groups
    prices : numpy.ndarray
        Array of prices (already filtered before passing)
    """
    mask = ~np.isnan(prices)
    prices_clean = prices[mask]
    groups_clean = neighbourhood_groups[mask]
    
    unique_groups = np.unique(groups_clean)

    data_by_group = [prices_clean[groups_clean == group] for group in unique_groups]
    
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.violinplot(data=data_by_group, inner='box', linewidth=1.5, ax=ax)
    
    for i, group_data in enumerate(data_by_group):
        mean_val = np.mean(group_data)
        ax.plot([i - 0.3, i + 0.3], [mean_val, mean_val], 
               color='darkblue', linestyle='--', linewidth=1.5, 
               label='Mean' if i == 0 else '')
    
    ax.set_xticks(np.arange(len(unique_groups)))
    ax.set_xticklabels(unique_groups)
    ax.set_xlabel('Neighbourhood Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title('Price Density Distribution by Neighbourhood', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_geographical_distribution(lon_data, lat_data, color_data=None, 
                                   title='Geographical Distribution'):
    """
    Plot geographical scatter plot
    
    Parameters:
    -----------
    lon_data : numpy.ndarray
        Array of longitude values
    lat_data : numpy.ndarray
        Array of latitude values
    color_data : numpy.ndarray
        Data array for coloring (optional, already filtered before passing)
    title : str
        Plot title
    """
    if color_data is not None:
        mask = (~np.isnan(lon_data)) & (~np.isnan(lat_data)) & (~np.isnan(color_data))
    else:
        mask = (~np.isnan(lon_data)) & (~np.isnan(lat_data))
    
    lon_clean = lon_data[mask]
    lat_clean = lat_data[mask]
    
    plt.figure(figsize=(12, 10))
    
    if color_data is not None:
        color_clean = color_data[mask]
        scatter = plt.scatter(lon_clean, lat_clean, c=color_clean, 
                            cmap=plt.get_cmap('jet'), alpha=0.6, s=15, vmin=0, vmax=500)
        plt.colorbar(scatter, label='Price')
    else:
        plt.scatter(lon_clean, lat_clean, color='blue', alpha=0.4, s=15)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_room_type_by_neighbourhood(neighbourhood_groups, room_types, prices):
    """
    Plot bar chart of room type distribution (Synchronized with Seaborn Countplot)
    """
    mask = ~np.isnan(prices)
    neighbourhood_clean = neighbourhood_groups[mask]
    room_types_clean = room_types[mask]
    
    unique_neighbourhoods = np.unique(neighbourhood_clean)
    unique_room_types = np.unique(room_types_clean)
    
    fig, axes = plt.subplots(1, len(unique_neighbourhoods), figsize=(20, 5))
    
    palette = sns.color_palette("muted")
    
    for idx, neighbourhood in enumerate(unique_neighbourhoods):
        ax = axes[idx]

        current_rooms = room_types_clean[neighbourhood_clean == neighbourhood]
        
        sns.countplot(x=current_rooms, ax=ax, order=unique_room_types, palette=palette)
        
        ax.set_title(neighbourhood)
        ax.set_xlabel('')
        ax.set_ylabel('Count' if idx == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        
        # Add count on top of bars
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.text(p.get_x() + p.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle('Room Type Distribution by Neighbourhood', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()


def plot_top_expensive_neighbourhoods(neighbourhoods, prices, top_n=10, most_expensive=True):
    """
    Plot top neighbourhoods (Using Seaborn Barplot)
    """
    mask = (~np.isnan(prices)) & (prices > 0)
    prices_clean = prices[mask]
    neighbourhoods_clean = neighbourhoods[mask]
    
    unique_neighbourhoods = np.unique(neighbourhoods_clean)
    
    avg_prices = []
    for neighbourhood in unique_neighbourhoods:
        mask_neigh = neighbourhoods_clean == neighbourhood
        avg_price = np.mean(prices_clean[mask_neigh])
        avg_prices.append(avg_price)
    
    avg_prices = np.array(avg_prices)
    
    if most_expensive:
        sorted_indices = np.argsort(avg_prices)[::-1][:top_n]
        title = f'Top {top_n} Most Expensive Neighbourhoods'
        palette = 'Reds_r'
    else:
        sorted_indices = np.argsort(avg_prices)[:top_n]
        title = f'Top {top_n} Cheapest Neighbourhoods'
        palette = 'Greens_r'
    
    top_neighbourhoods = unique_neighbourhoods[sorted_indices]
    top_prices = avg_prices[sorted_indices]
    
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(x=top_prices, y=top_neighbourhoods, hue=top_neighbourhoods, palette=palette, legend=False)
    
    ax.set_xlabel('Average Price ($)')
    ax.set_title(title)
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 1, p.get_y() + p.get_height()/2,
                f'${top_prices[i]:.3f}',
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_price_distribution_by_room_type(room_types, prices, max_price=500):
    """
    Plot price distribution by room type (Seaborn Violinplot)
    """
    mask = (~np.isnan(prices)) & (prices <= max_price) & (prices > 0)
    prices_clean = prices[mask]
    room_types_clean = room_types[mask]
    unique_types = np.unique(room_types_clean)
    
    data_to_plot = [prices_clean[room_types_clean == rt] for rt in unique_types]
        
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.violinplot(data=data_to_plot, palette='pastel', inner='box', linewidth=1.5, ax=ax)

    for i, group_data in enumerate(data_to_plot):
        mean_val = np.mean(group_data)
        ax.plot([i - 0.3, i + 0.3], [mean_val, mean_val], 
               color='darkblue', linestyle='--', linewidth=1.5, 
               label='Mean' if i == 0 else '')
    
    ax.set_xticks(np.arange(len(unique_types)))
    ax.set_xticklabels(unique_types)
    ax.set_xlabel('Room Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Price Distribution by Room Type', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_top_hosts(host_ids, host_names, top_n=15):
    """
    Plot top hosts (Using Seaborn Barplot)
    """
    unique_hosts, counts = np.unique(host_ids, return_counts=True)
    
    sorted_indices = np.argsort(counts)[::-1][:top_n]
    top_host_ids = unique_hosts[sorted_indices]
    top_counts = counts[sorted_indices]
    
    top_names = []
    for hid in top_host_ids:
        mask = host_ids == hid
        name = host_names[mask][0]
        top_names.append(name)
    
    plt.figure(figsize=(12, 8))

    ax = sns.barplot(x=top_counts, y=top_names, hue=top_names, palette="Oranges_r", legend=False)
    
    ax.set_xlabel('Number of Listings')
    ax.set_title(f'Top {top_n} Hosts with Most Listings')
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 0.5, p.get_y() + p.get_height()/2,
                f'{int(top_counts[i])}',
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_top_words_name(data, column_names, top_n=10):
    """
    Standardized function: Using strong cleaning logic (isalnum) from function A 
    and beautiful Seaborn interface from function B.
    """
    # 1. Get data from name column
    if 'name' not in column_names:
        print("Column 'name' not found.")
        return
    col_idx = np.where(column_names == 'name')[0][0]
    names_data = data[:, col_idx]

    # 2. Define Stopwords (Complete combination)
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'near',
        'it', 'this', 'that', 'these', 'those', 'if', 'so', 'my', 'very', 
        'your'
    }
    # Note: If you want to count the word 'room', remove it from the set above.
    
    word_counts = {}
    
    # 3. Process word counting (Optimized logic from Function A)
    for name in names_data:
        # Handle NaN/None safely
        s_name = str(name)
        if s_name.lower() in ['nan', 'none', '']:
            continue
            
        # Convert to lowercase
        name_lower = s_name.lower()
        
        # Thorough cleaning: Keep only letters and numbers, replace rest with spaces
        # This removes emojis and special characters
        name_clean = ""
        for char in name_lower:
            if char.isalnum():
                name_clean += char
            else:
                name_clean += " "
        
        # Split words
        words = name_clean.split()
        
        for w in words:
            if len(w) > 1 and w not in stopwords:
                word_counts[w] = word_counts.get(w, 0) + 1
                
    # 4. Sort Top N
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    top_list = sorted_words[:top_n]
    
    if not top_list:
        print("No words found.")
        return
        
    labels = [item[0] for item in top_list]
    values = [item[1] for item in top_list]
    
    # 5. Plot chart (Synchronized Seaborn style)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    
    # Plot
    sns.barplot(x=labels, y=values, hue=labels, palette='viridis', legend=False, ax=ax)
    
    ax.set_title(f'Top {top_n} Most Common Words in Listing Names (Thoroughly Filtered)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Words', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    # Add values
    max_val = max(values)
    ax.set_ylim(0, max_val * 1.15)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + (max_val * 0.01),
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    plt.show()