import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    
    Parameters:
    -----------
    data : numpy.ndarray
        The dataset containing the data
    column_names : numpy.ndarray
        Array of column names
    target_columns : list
        List of columns to check for missing values
    title : str
        Title of the plot
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
    
    Parameters:
    -----------
    data : numpy.ndarray
        Array of data to plot
    column_name : str
        Name of the column being plotted
    bins : int
        Number of bins for the histogram
    title : str
        Title of the plot
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
    
    Parameters:
    -----------
    data : numpy.ndarray
        Array of categorical data
    column_name : str
        Name of the column being plotted
    top_n : int
        Number of top categories to display
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
    
    Parameters:
    -----------
    data : numpy.ndarray
        The dataset for correlation calculation
    column_names : list
        List of column names to include in the heatmap
    figsize : tuple
        Size of the figure
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
    
    Parameters:
    -----------
    categories : numpy.ndarray
        Array of category labels
    prices : numpy.ndarray
        Array of prices corresponding to categories
    category_name : str
        Name of the category
    top_n : int
        Number of top categories to display
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
    
    Parameters:
    -----------
    neighbourhood_groups : numpy.ndarray
        Array of neighbourhood groups
    room_types : numpy.ndarray
        Array of room types
    prices : numpy.ndarray
        Array of prices (used for filtering valid data)
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
    
    Parameters:
    -----------
    neighbourhoods : numpy.ndarray
        Array of neighbourhood names
    prices : numpy.ndarray
        Array of prices
    top_n : int
        Number of neighbourhoods to display
    most_expensive : bool
        If True, plot most expensive; otherwise, plot cheapest
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
    
    Parameters:
    -----------
    room_types : numpy.ndarray
        Array of room types
    prices : numpy.ndarray
        Array of prices
    max_price : int
        Maximum price to include in the plot
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
    
    Parameters:
    -----------
    host_ids : numpy.ndarray
        Array of host IDs
    host_names : numpy.ndarray
        Array of host names
    top_n : int
        Number of top hosts to display
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
    
    Parameters:
    -----------
    data : numpy.ndarray
        The dataset containing the data
    column_names : numpy.ndarray
        Array of column names
    top_n : int
        Number of top words to display
    """
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
    
    word_counts = {}
    
    for name in names_data:
        s_name = str(name)
        if s_name.lower() in ['']:
            continue
            
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
                
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    top_list = sorted_words[:top_n]
    
    if not top_list:
        print("No words found.")
        return
        
    labels = [item[0] for item in top_list]
    values = [item[1] for item in top_list]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    
    # Plot
    sns.barplot(x=labels, y=values, hue=labels, palette='viridis', legend=False, ax=ax)
    
    ax.set_title(f'Top {top_n} Most Common Words in Listing Names (Thoroughly Filtered)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Words', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    max_val = max(values)
    ax.set_ylim(0, max_val * 1.15)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + (max_val * 0.01),
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    plt.show()


def plot_business_efficiency_comparison(top_15_avg, individual_avg):
    """
    Plot business efficiency comparison between Top 15 hosts and individual hosts
    
    Parameters:
    -----------
    top_15_avg : float
        Average reviews per month for Top 15 hosts
    individual_avg : float
        Average reviews per month for individual hosts
    """
    categories = ['Top 15 Hosts\n(Large Portfolio)', 'Individual Hosts\n(Single Listing)']
    averages = [top_15_avg, individual_avg]
    
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(x=categories, y=averages, hue=categories, 
                     palette=['#FF5A5F', '#00A699'], legend=False)
    
    ax.set_ylabel('Average Reviews per Month', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Business Efficiency: Reviews per Month Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height,
                f'{averages[i]:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_pricing_strategy_comparison(top_15_avg_price, market_avg_price):
    """
    Plot pricing strategy comparison between Top 15 hosts and market average
    
    Parameters:
    -----------
    top_15_avg_price : float
        Average price for Top 15 hosts
    market_avg_price : float
        Average price for other hosts
    """
    categories = ['Top 15 Hosts', 'Other Hosts\n(Market Average)']
    avg_prices = [top_15_avg_price, market_avg_price]
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(x=categories, y=avg_prices, hue=categories,
                     palette=['#FF5A5F', '#767676'], legend=False)
    
    ax.set_ylabel('Average Price ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Pricing Strategy: Top Hosts vs. Market Average', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height,
                f'${avg_prices[i]:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    price_diff_pct = ((top_15_avg_price - market_avg_price) / market_avg_price * 100)
    max_price = max(avg_prices)
    ax.annotate(f'{price_diff_pct:+.1f}%',
                xy=(0.5, max_price * 1.05),
                fontsize=14, fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def plot_minimum_nights_comparison(top_15_avg_min_nights, market_avg_min_nights,
                                   top_15_dist, market_dist, stay_categories):
    """
    Plot minimum nights comparison with distribution breakdown
    
    Parameters:
    -----------
    top_15_avg_min_nights : float
        Average minimum nights for Top 15 hosts
    market_avg_min_nights : float
        Average minimum nights for market
    top_15_dist : list
        Distribution percentages for Top 15 hosts [short, medium, long]
    market_dist : list
        Distribution percentages for market [short, medium, long]
    stay_categories : list
        Category labels for stay lengths
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    categories = ['Top 15 Hosts', 'Other Hosts']
    avg_mins = [top_15_avg_min_nights, market_avg_min_nights]
    
    sns.barplot(x=categories, y=avg_mins, hue=categories,
                palette=['#FF5A5F', '#767676'], legend=False, ax=ax1)
    
    ax1.set_ylabel('Average Minimum Nights', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_title('Minimum Stay Requirements Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, p in enumerate(ax1.patches):
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width()/2., height,
                 f'{avg_mins[i]:.1f} nights',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    import pandas as pd
    df_dist = pd.DataFrame({
        'Category': stay_categories * 2,
        'Percentage': top_15_dist + market_dist,
        'Host Type': ['Top 15 Hosts'] * len(stay_categories) + ['Other Hosts'] * len(stay_categories)
    })
    
    sns.barplot(data=df_dist, x='Category', y='Percentage', hue='Host Type',
                palette=['#FF5A5F', '#767676'], ax=ax2)
    
    ax2.set_ylabel('Percentage of Listings (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_title('Stay Length Distribution', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=10, title='')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for p in ax2.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax2.text(p.get_x() + p.get_width()/2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_market_domination(sorted_neighbourhoods, ng_names, ng_concentrations, ng_top15, ng_totals):
    """
    Plot market domination analysis by neighbourhood and neighbourhood group
    
    Parameters:
    -----------
    sorted_neighbourhoods : list of tuples
        List of (neighbourhood_name, stats_dict) sorted by concentration
    ng_names : list
        Neighbourhood group names
    ng_concentrations : list
        Concentration percentages for each neighbourhood group
    ng_top15 : list
        Number of Top 15 listings in each neighbourhood group
    ng_totals : list
        Total listings in each neighbourhood group
    """
    top_10_names = [item[0] for item in sorted_neighbourhoods[:10]]
    top_10_concentrations = [item[1]['concentration'] for item in sorted_neighbourhoods[:10]]
    top_10_totals = [item[1]['total'] for item in sorted_neighbourhoods[:10]]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    sns.barplot(x=top_10_concentrations, y=top_10_names, hue=top_10_names,
                palette='Reds_r', legend=False, ax=ax1)
    
    ax1.set_xlabel('Top 15 Hosts Concentration (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('')
    ax1.set_title('Top 10 Most Dominated Neighbourhoods by Top 15 Hosts', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    for i, p in enumerate(ax1.patches):
        width = p.get_width()
        ax1.text(width + 1, p.get_y() + p.get_height()/2,
                 f'{top_10_concentrations[i]:.1f}% ({sorted_neighbourhoods[i][1]["top_15"]}/{top_10_totals[i]})',
                 ha='left', va='center', fontsize=10, fontweight='bold')

    colors_ng = ['#FF5A5F', '#00A699', '#FC642D', '#484848', '#767676'][:len(ng_names)]
    
    sns.barplot(x=ng_names, y=ng_concentrations, hue=ng_names,
                palette=colors_ng, legend=False, ax=ax2)
    
    ax2.set_ylabel('Top 15 Hosts Concentration (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_title('Market Domination by Neighbourhood Group', fontsize=14, fontweight='bold', pad=20)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for i, p in enumerate(ax2.patches):
        height = p.get_height()
        ax2.text(p.get_x() + p.get_width()/2., height,
                 f'{ng_concentrations[i]:.1f}%\n({ng_top15[i]}/{ng_totals[i]})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_neighbourhood_price_analysis(analysis_data, neighbourhoods_order, top_2_expensive, top_2_cheapest):
    """
    Plot detailed analysis of top 2 most expensive and cheapest neighbourhoods
    including price comparison, room type distribution, and stay length distribution
    
    Parameters:
    -----------
    analysis_data : dict
        Dictionary containing analysis data for each neighbourhood with keys:
        - 'avg_price': average price
        - 'room_types': dict of room type percentages
        - 'stay_dist': dict of stay length percentages
        - 'avg_min_nights': average minimum nights
    neighbourhoods_order : list
        List of neighbourhood names in order [expensive1, expensive2, cheap1, cheap2]
    top_2_expensive : list
        List of 2 most expensive neighbourhood names
    top_2_cheapest : list
        List of 2 cheapest neighbourhood names
    """
    
    expensive_color = '#FF6B6B'
    cheap_color = '#4ECDC4'
    
    colors_list = [expensive_color, expensive_color, cheap_color, cheap_color]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    prices_list = [analysis_data[n]['avg_price'] for n in neighbourhoods_order]
    
    sns.barplot(x=neighbourhoods_order, y=prices_list, hue=neighbourhoods_order,
                palette=colors_list, legend=False, ax=ax1)
    
    for i, p in enumerate(ax1.patches):
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width()/2., height,
                 f'${prices_list[i]:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Average Price ($)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_title('Average Price Comparison: Most Expensive vs. Cheapest Neighbourhoods', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(prices_list) * 1.15)

    legend_elements = [Patch(facecolor=expensive_color, label='Most Expensive'),
                       Patch(facecolor=cheap_color, label='Cheapest')]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

    room_type_order = ['Entire home/apt', 'Private room', 'Shared room']
    room_type_colors = {'Entire home/apt': '#FF6B6B', 'Private room': '#4ECDC4', 'Shared room': '#95E1D3'}
    
    for idx, neighbourhood in enumerate(neighbourhoods_order):
        row = 1 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        room_data = analysis_data[neighbourhood]['room_types']

        types = []
        percentages = []
        colors = []
        
        for rt in room_type_order:
            if rt in room_data:
                types.append(rt)
                percentages.append(room_data[rt])
                colors.append(room_type_colors[rt])
        
        bars = ax.barh(types, percentages, color=colors, alpha=0.8)

        for bar, pct in zip(bars, percentages):
            width = bar.get_width()
            if width > 0:
                ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{pct:.1f}%',
                        ha='left', va='center', fontsize=10, fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Percentage (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{neighbourhood}\nRoom Type Distribution', 
                     fontsize=11, fontweight='bold', 
                     color=expensive_color if idx < 2 else cheap_color)
        ax.set_xlim(0, 120)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

    fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    stay_colors = ['#FFD93D', '#6BCB77', '#4D96FF']
    
    for idx, neighbourhood in enumerate(neighbourhoods_order):
        ax = axes[idx]
        
        stay_data = analysis_data[neighbourhood]['stay_dist']
        
        categories = list(stay_data.keys())
        percentages = list(stay_data.values())

        sns.barplot(x=categories, y=percentages, hue=categories,
                    palette=stay_colors, legend=False, ax=ax)

        for i, p in enumerate(ax.patches):
            height = p.get_height()
            if height > 0:
                ax.text(p.get_x() + p.get_width()/2., height,
                        f'{percentages[i]:.1f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Percentage of Listings (%)', fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_title(f'{neighbourhood}\nMinimum Nights Distribution (Avg: {analysis_data[neighbourhood]["avg_min_nights"]:.1f} nights)', 
                     fontsize=11, fontweight='bold',
                     color=expensive_color if idx < 2 else cheap_color)
        ax.set_ylim(0, max(percentages) * 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
