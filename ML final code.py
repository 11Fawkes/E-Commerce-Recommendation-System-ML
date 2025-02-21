import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_parquet(r'C:\Users\dk103\Downloads\test.parquet\test.parquet')

sample_df = data.sample(n=1200, random_state=42)
sample_df.reset_index(drop=True, inplace=True)

sample_df['brand'] = sample_df['brand'].astype(str).str.lower()
sample_df['cat_0'] = sample_df['cat_0'].astype(str).str.lower()
sample_df['cat_1'] = sample_df['cat_1'].astype(str).str.lower()
sample_df['cat_2'] = sample_df['cat_2'].astype(str).str.lower()

sample_df['implicit_rating'] = sample_df['event_type'].map({'purchase': 2, 'cart': 1, 'view': 0.5})

interaction_data = sample_df.groupby(['user_id', 'product_id'])['implicit_rating'].sum().reset_index()
user_item_matrix = interaction_data.pivot(index='user_id', columns='product_id', values='implicit_rating').fillna(0)
user_item_sparse = csr_matrix(user_item_matrix.values)

num_factors = 10
U, sigma, Vt = svds(user_item_sparse, k=num_factors)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

relevant_cols = ['brand', 'cat_0', 'cat_1', 'cat_2']
sample_df['content'] = sample_df[relevant_cols].apply(lambda x: ' '.join(x), axis=1)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(sample_df['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_items_hybrid(keyword, df=sample_df, num_recs=5, content_weight=0.5, collab_weight=0.5, sort_by=None):
    relevant_items = df[df['content'].str.contains(keyword.lower())]['product_id'].unique()
    content_recs = []
    for item_id in relevant_items:
        idx = df.index[df['product_id'] == item_id][0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        content_recs.extend([(df['product_id'][i], score) for i, score in sim_scores[1:num_recs+1]])
    
    collab_recs = predicted_ratings_df.mean(axis=0).sort_values(ascending=False)[:num_recs].index.tolist()
    
    all_recs = content_recs + [(rec, 1.0) for rec in collab_recs]
    scored_recs = {}
    for item, score in all_recs:
        scored_recs[item] = scored_recs.get(item, 0) + score * content_weight
        if item in collab_recs:
            scored_recs[item] += collab_weight
    
    top_recs = sorted(scored_recs.items(), key=lambda x: x[1], reverse=True)[:num_recs]
    product_details = df[df['product_id'].isin([rec[0] for rec in top_recs])][['product_id', 'brand', 'price', 'cat_0', 'cat_1', 'cat_2']]
    
    if sort_by == 'price_asc':
        product_details = product_details.sort_values(by='price')
    elif sort_by == 'price_desc':
        product_details = product_details.sort_values(by='price', ascending=False)
    elif sort_by == 'most_bought':
        product_counts = df[df['event_type'] == 'purchase'].groupby('product_id').size()
        product_details = product_details.join(product_counts.rename('purchase_count'), on='product_id').sort_values(by='purchase_count', ascending=False).drop(columns='purchase_count')
    elif sort_by == 'least_bought':
        product_counts = df[df['event_type'] == 'purchase'].groupby('product_id').size()
        product_details = product_details.join(product_counts.rename('purchase_count'), on='product_id').sort_values(by='purchase_count').drop(columns='purchase_count')
    
    return product_details.head(num_recs).to_dict(orient='records')

def user_interaction():
    while True:
        print("\nOptions:")
        print("1. Get recommendations based on a search keyword")
        print("2. Exit")
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == '1':
            keyword = input("Enter your search keyword: ")
            try:
                num_recs = int(input("Enter the number of recommendations you want (e.g., 5, 10, 15): "))
                sort_by = input("Enter the sorting method (price_asc, price_desc, most_bought, least_bought): ").strip().lower()
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
            recommendations = recommend_items_hybrid(keyword, num_recs=num_recs, sort_by=sort_by)
            if recommendations:
                print(f"\nTop {num_recs} Recommendations based on '{keyword}' sorted by {sort_by}:")
                for i, rec in enumerate(recommendations):
                    print(f"\nRecommendation {i+1}:")
                    print(f"  Product ID: {rec['product_id']}")
                    print(f"  Brand: {rec['brand']}")
                    print(f"  Price: {rec['price']}")
                    print(f"  Category 0: {rec['cat_0']}")
                    print(f"  Category 1: {rec['cat_1']}")
                    print(f"  Category 2: {rec['cat_2']}")
            else:
                print(f"No recommendations found for the keyword '{keyword}'.")
        elif choice == '2':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

user_interaction()
