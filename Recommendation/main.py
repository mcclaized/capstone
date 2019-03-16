from data import SingleDayDataLoader
from model import similar_bonds_pipeline

# Load once in the pre-forked server process for efficiency
data_loader = SingleDayDataLoader()

def get_similar_bonds(isin, cohort_attributes, features):
    
    # ISIN to Pandas Series of data for that bond
    bond = data_loader.get_bond(isin)

    # Pandas Series to Pandas DataFrame of data for all bonds in the specified cohort
    bond_cohort = data_loader.get_cohort(bond, attributes=cohort_attributes)
    bond_cohort = bond_cohort[features]
   
    # Fit the model
    model = similar_bonds_pipeline()
    model.fit(bond_cohort)

    # Find similar bonds
    k_neighbors = min(bond_cohort.shape[0], 10)
    distances, indices = model.predict(bond[features], k_neighbors=k_neighbors)
    #import pdb;pdb.set_trace()
    similar_bond_isins = bond_cohort.iloc[indices.ravel()].index.values
    # Exclude the input isin from the list of similar bonds
    similar_bond_isins = [i for i in similar_bond_isins if i != isin]
    similar_bonds = data_loader.get_bonds(similar_bond_isins)
    
    return similar_bonds

if __name__ == '__main__':
    from tabulate import tabulate

    cohort_attributes = ['BCLASS3', 'Country', 'Ticker', 'Class - Detail - Code']
    features = ["OAS", "OAD", "KRD 5Y", "KRD 10Y", "KRD 20Y", "KRD 30Y"]

    while True:
        isin = input("\n\nPlease Enter an ISIN: ")
        bond = data_loader.get_bond(isin)
        bond_table = tabulate(bond[cohort_attributes + features], headers='keys', tablefmt='psql')
        print("\nSearching for bonds that are similar to these characteristics:\n{}".format(bond_table))
        similar_bonds = get_similar_bonds(isin, cohort_attributes=cohort_attributes, features=features)
        similar_bonds_table = tabulate(similar_bonds[cohort_attributes + features], headers='keys', tablefmt='psql')
        print("\nHere are your similar bonds!\n{}\n".format(similar_bonds_table))

