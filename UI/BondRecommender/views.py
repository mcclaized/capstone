from django.shortcuts import render
from BondRecommender.data_loader import SingleDayDataLoader
from BondRecommender.recommendation_models import similar_bonds_pipeline
from django.http import HttpResponse

# Create your views here.
def home(request):
    context = dict()
    return render(request, 'home.html', context)

def results(request):
    # Load once in the pre-forked server process for efficiency
    single_day_data = SingleDayDataLoader()

    context = dict()
    sec_id = str(request.GET['sec_id'])
    context['sec_id'] = sec_id
    context['sec_id_error'] = 0

    try:
        ## COLUMNS WHOSE VALUES MUST BE THE SAME IN ORDER TO BE CONSIDERED SIMILAR
        # cohort_attributes = ['BCLASS3', 'Country', 'Ticker', 'Class - Detail - Code']
        cohort_attributes = None

        ## COLUMNS THAT THE MODEL SHOULD CONSIDER WHEN LOOKING FOR SIMILAR BONDS
        features = ["OAS", "OAD", "KRD 5Y", "KRD 10Y", "KRD 20Y", "KRD 30Y"]

        ## COLUMNS TO DISPLAY IN THE CLI OUTPUT
        display_columns = ['ISIN', 'Ticker', 'BCLASS3', 'Country'] + (features or [])
        context['display_columns'] = display_columns

        bond = single_day_data.get_bond(sec_id)
        bond = bond.reset_index()
        bond_table = bond[display_columns]
        context['bond_table'] = bond_table.values.tolist()

        similar_bonds = get_similar_bonds(sec_id, single_day_data, features=features, cohort_attributes=cohort_attributes)
        similar_bonds = similar_bonds.reset_index()
        similar_bonds_table = similar_bonds[display_columns].values.tolist()
        context['similar_bonds_table'] = similar_bonds_table

        return render(request, 'results.html', context)

    except  KeyError:
        context['sec_id_error'] = 1
        return render(request, 'home.html', context)

def get_similar_bonds(isin, single_day_data, features=None, cohort_attributes=None):
    """
    This is a top-level function that is meant to be called when processing a server requst for SimilarBonds

    :param isin:                The ISIN identifier of the bond we're trying to find similar bonds for
    :param features:            Optional. A list of columns to consider when determining bond similarity.
                                Default: None, meaning all columns in the data set
    :param cohort_attributes:   Optional. A list of columns specifying the bond attributes that *must* be the same in order for a bond to be considered similar
                                Default: None, meaning all bonds are valid candidates
    """

    # ISIN to Pandas Series of data for that bond
    bond = single_day_data.get_bond(isin)

    # Pandas Series to Pandas DataFrame of data for all bonds in the specified cohort
    if cohort_attributes is None:
        cohort_attributes = []
        bond_cohort = single_day_data.data
    else:
        bond_cohort = single_day_data.get_cohort(bond, attributes=cohort_attributes)

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

    similar_bonds = single_day_data.get_bonds(similar_bond_isins)

    return similar_bonds

def view_plot_oas(request):
    from BondRecommender.data_loader import MultipleDayDataLoader
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import matplotlib.pyplot as plt
    import io

    multi_data_loader = MultipleDayDataLoader()
    isin = 'US06406RAC16'
    bond = multi_data_loader.get_bond(isin)

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(bond["date"].values, bond["OAS"].values, color='red')
    ax1.set_ylabel("OAS")
    ax1.set_xlabel("date")
    ax1.set_title('OAS spread 30 days for ' + str(isin))

    FigureCanvas(fig)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response