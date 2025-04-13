import pandas as pd

# Load the two CSVs
df_top3 = pd.read_csv("ssl_to_include_top3_mapping.csv")
df_sim = pd.read_csv("similarity_metrics_gloss_pairs_new.csv")

# Optional: drop duplicates in gloss2 so we avoid one-to-many joins
# Keep the first match (you can customize this if needed)
df_sim_unique = df_sim.drop_duplicates(subset='gloss2')

# Merge with left join to preserve the shape of df_top3
merged_df = pd.merge(
    df_top3,
    df_sim_unique[['gloss2', 'video2']],
    left_on='predicted_include_gloss',
    right_on='gloss2',
    how='left'
)

# Drop the now-redundant 'gloss2' column
merged_df.drop(columns=['gloss2'], inplace=True)

# Save the final merged result
merged_df.to_csv("ssl_to_include_top3_mapping_full.csv", index=False)
