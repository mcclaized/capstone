from BondRecommender.data_loader import SingleDayDataLoader
from BondRecommender.recommendation_models import similar_bonds_pipeline

# Load once in the pre-forked server process for efficiency
data_loader = SingleDayDataLoader()

def get_similar_bonds(isin, features=None, cohort_attributes=None):
    """
    This is a top-level function that is meant to be called when processing a server requst for SimilarBonds

    :param isin:                The ISIN identifier of the bond we're trying to find similar bonds for
    :param features:            Optional. A list of columns to consider when determining bond similarity. 
                                Default: None, meaning all columns in the data set
    :param cohort_attributes:   Optional. A list of columns specifying the bond attributes that *must* be the same in order for a bond to be considered similar
                                Default: None, meaning all bonds are valid candidates
    """
    
    # ISIN to Pandas Series of data for that bond
    bond = data_loader.get_bond(isin)

    # Pandas Series to Pandas DataFrame of data for all bonds in the specified cohort
    if cohort_attributes is None:
        cohort_attributes = []
        bond_cohort = data_loader.data
    else:
        bond_cohort = data_loader.get_cohort(bond, attributes=cohort_attributes)
    
    if features is None:
        features = [col for col in bond_cohort.columns if col not in cohort_attributes]
  
    # Fit the model
    model = similar_bonds_pipeline()
    model.fit(bond_cohort[features])

    # Find similar bonds
    k_neighbors = min(bond_cohort.shape[0], 10)
    distances, indices = model.predict(bond[features], k_neighbors=k_neighbors)
    similar_bond_isins = bond_cohort.iloc[indices.ravel()].index.values
    # Exclude the input isin from the list of similar bonds
    similar_bond_isins = [i for i in similar_bond_isins if i != isin]

    similar_bonds = data_loader.get_bonds(similar_bond_isins)
    
    return similar_bonds

if __name__ == '__main__':
    from tabulate import tabulate

    ## COLUMNS WHOSE VALUES MUST BE THE SAME IN ORDER TO BE CONSIDERED SIMILAR
    #cohort_attributes = ['BCLASS3', 'Country', 'Ticker', 'Class - Detail - Code']
    cohort_attributes = None

    ## COLUMNS THAT THE MODEL SHOULD CONSIDER WHEN LOOKING FOR SIMILAR BONDS
    features = ["OAS", "OAD", "KRD 5Y", "KRD 10Y", "KRD 20Y", "KRD 30Y"]

    ## COLUMNS TO DISPLAY IN THE CLI OUTPUT
    display_columns = ['Ticker', 'BCLASS3', 'Country']  + (features or [])

    while True:
        isin = input("\n\nPlease Enter an ISIN: ")
        try:
            bond = data_loader.get_bond(isin)
        except KeyError:
            print("Invalid ISIN! Please try again.")
            continue
        bond_table = tabulate(bond[display_columns], headers='keys', tablefmt='psql')
        print("\nSearching for bonds that are similar to these characteristics:\n{}".format(bond_table))

        similar_bonds = get_similar_bonds(isin, features=features, cohort_attributes=cohort_attributes)
        similar_bonds_table = tabulate(similar_bonds[display_columns], headers='keys', tablefmt='psql')
        print("\nHere are your similar bonds!\n{}\n".format(similar_bonds_table))

