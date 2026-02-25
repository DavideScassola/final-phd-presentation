import pandas as pd

df_sampling_time = pd.read_csv('/home/davide/PhD/phd_thesis/tables/csv/sampling_time.csv')

# Filter for Average dataset only
df_avg = df_sampling_time[df_sampling_time['dataset'] == 'Average'].copy()

# Get TabPC sampling time
tabpc_time = df_avg[df_avg['method'] == 'TabPC']['mean'].values[0]

# Calculate speedup for all other methods
df_avg['speedup'] = df_avg['mean'] / tabpc_time

# Display results
print("Average Sampling Time Speedup - TabPC vs Other Methods:")
print("=" * 60)
for idx, row in df_avg.iterrows():
    if row['method'] != 'TabPC':
        tabpc_speedup = row['speedup']  # how many times slower this method is
        print(f"TabPC is {tabpc_speedup:>8.2f}x faster than {row['method']:<20}")

print("\n" + "=" * 60)
avg_speedup = df_avg[df_avg['method'] != 'TabPC']['speedup'].mean()
print(f"TabPC is on average {avg_speedup:.2f}x faster than other methods")

