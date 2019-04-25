from BondRecommender.models import Securities, vanguard_merge
from BondRecommender.data_loader import SingleDayDataLoader
from BondRecommender.data_loader import MultipleDayDataLoader
from BondRecommender.recommendation_models import similar_bonds_pipeline
from BondRecommender.prediction_models import predict_rc
from django.http import HttpResponse
from django.shortcuts import render

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
    numofdays = int(request.GET['num_of_days'])
    numofrecomm = int(request.GET['num_of_recommendation'])
    cohort_filtering = str(request.GET['cohort_filtering'])
    ytm_upper = str(request.GET['ytm_upper'])
    ytm_lower = str(request.GET['ytm_lower'])
    oad_upper = str(request.GET['oad_upper'])
    oad_lower = str(request.GET['oad_lower'])

    try:
        ## COLUMNS WHOSE VALUES MUST BE THE SAME IN ORDER TO BE CONSIDERED SIMILAR
        if cohort_filtering == "Yes":
            cohort_attributes = ['BCLASS3',  'Ticker', 'Country', 'Class - Detail - Code']
        else:
            cohort_attributes = None

        ## COLUMNS THAT THE MODEL SHOULD CONSIDER WHEN LOOKING FOR SIMILAR BONDS
        features = ["OAS", "OAD", "KRD 5Y", "KRD 10Y", "KRD 20Y", "KRD 30Y"]

        ## COLUMNS TO DISPLAY IN THE CLI OUTPUT
        display_columns = ['ISIN', 'Ticker', 'BCLASS3', 'Country'] + (features or []) + ['Yield to Mat', 'Cpn', 'Px Close']
        #context['display_columns'] = display_columns

        bond = single_day_data.get_bond(sec_id)

        filter_conditions = {'Yield to Mat': (ytm_lower, ytm_upper), 'OAD': (oad_lower, oad_upper)}

        bond = bond.reset_index()
        bond_table = bond[display_columns]
        context['bond_table'] = bond_table.values.tolist()

        similar_bonds = get_similar_bonds(sec_id, single_day_data, numofrecomm+1, filter_conditions, features=features, cohort_attributes=cohort_attributes)
        similar_bonds = similar_bonds.reset_index()
        similar_bonds_table = similar_bonds[display_columns].values.tolist()
        #context['similar_bonds_table'] = similar_bonds_table

        multi_data_loader = MultipleDayDataLoader(numdays=numofdays)
        bond = multi_data_loader.get_bond(sec_id)
        context['bond_dates'] = '|'.join(bond["date"].values)
        context['bond_OAS'] = '|'.join(list(map(str, bond["OAS"].values)))

        ### JPM
        isins = similar_bonds.ISIN.unique()
        prediction_result = predict_rc(multi_data_loader, date=None, isins=isins)
        result = similar_bonds.merge(prediction_result, on="ISIN", how ="outer")
        display_columns = display_columns + ["rich/cheap"]
        result_table = result[display_columns].values.tolist()
        context["similar_bonds_table"] = result_table
        context['display_columns'] = display_columns

        # for sb in similar_bonds_table:
        #     sb_isin = str(sb[0])
        #     bond = multi_data_loader.get_bond(sb_isin)
        #     context[sb_isin + '_dates'] = '|'.join(bond["date"].values)
        #     context[sb_isin + '_OAS'] = '|'.join(list(map(str, bond["OAS"].values)))

        # bond_dates2 = dict()
        # bond_OAS2 = dict()
        #
        # for l in context['similar_bonds_table']:
        #     sec_code = l[0]
        #     bond = multi_data_loader.get_bond(sec_code)
        #     bond_dates2.update({sec_code: '|'.join(bond["date"].values)})
        #     bond_OAS2.update({sec_code: '|'.join(list(map(str, bond["OAS"].values)))})
        #
        # context['bond_dates2'] = bond_dates2
        # context['bond_OAS2'] = bond_OAS2

        return render(request, 'results.html', context)

    except KeyError:
        context['sec_id_error'] = 1
        return render(request, 'home.html', context)

def get_similar_bonds(isin, single_day_data, num_of_bonds, filter_conditions, features=None, cohort_attributes=None):
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

    ytm_lower = filter_conditions['Yield to Mat'][0]
    ytm_upper = filter_conditions['Yield to Mat'][1]
    oad_lower = filter_conditions['OAD'][0]
    oad_upper = filter_conditions['OAD'][1]

    # Filter by ytm range
    if (ytm_upper != '') and (ytm_lower != ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'].between(float(ytm_lower), float(ytm_upper), inclusive=True)]
    elif (ytm_upper != '') and (ytm_lower == ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'] < float(ytm_upper)]
    elif (ytm_upper == '') and (ytm_lower != ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'] > float(ytm_lower)]
    else:
        pass

    # Filter by oad range
    if (oad_upper != '') and (oad_lower != ''):
        bond_cohort = bond_cohort[bond_cohort['OAD'].between(float(oad_lower), float(oad_upper), inclusive=True)]
    elif (oad_upper != '') and (oad_lower == ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'] < float(oad_upper)]
    elif (oad_upper == '') and (oad_lower != ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'] > float(oad_lower)]
    else:
        pass

    if features is None:
        features = [col for col in bond_cohort.columns if col not in cohort_attributes]

    # Fit the model
    model = similar_bonds_pipeline()
    model.fit(bond_cohort[features])

    # Find similar bonds
    k_neighbors = min(bond_cohort.shape[0], num_of_bonds)
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
    isin2 = 'US857477AZ63'
    bond = multi_data_loader.get_bond(isin)
    bond2 = multi_data_loader.get_bond(isin2)

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(bond["date"].values, bond["OAS"].values, color='red')
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.plot(bond2["date"].values, bond2["OAS"].values, color='blue')
    ax1.set_ylabel("OAS")
    ax1.set_xlabel("date")
    ax1.set_title('OAS spread 30 days for ' + str(isin))

    FigureCanvas(fig)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response

def view_graph(request):
    context = dict()
    sec_id = 'US857477AZ63'#str(request.GET['sec_id'])
    context['sec_id'] = sec_id
    numofdays = 30#int(request.GET['num_of_days'])

    multi_data_loader = MultipleDayDataLoader(numdays=numofdays)
    bond = multi_data_loader.get_bond(sec_id)
    context['bond_dates'] = '|'.join(bond["date"].values)
    context['bond_OAS'] = '|'.join(list(map(str, bond["OAS"].values)))

    bond2 = multi_data_loader.get_bond('US857477AZ63')
    context['bond_dates2'] = '|'.join(bond2["date"].values)
    context['bond_OAS2'] = '|'.join(list(map(str, bond2["OAS"].values)))

    return render(request, 'results.html', context)

# ### old ###
#
# def get_securities():
#     import xlrd
#
#     workbook = xlrd.open_workbook('./Data/SampleData.xlsx')
#     worksheet = workbook.sheet_by_index(0)
#
#     for row in range(1, worksheet.nrows):
#         #for col in range(worksheet.ncols):
#         try:
#             s = Securities.objects.get(isin=str(worksheet.cell_value(row, 0)))
#         # if there isn’t, create a new entry in the Securities table with the appropriate data
#         except:
#             try:
#                 s = Securities(isin=str(worksheet.cell_value(row, 0)),
#                                YAS_price=float(worksheet.cell_value(row, 1)),
#                                OAS_spread=str(worksheet.cell_value(row, 2)),
#                                modified_duration=str(worksheet.cell_value(row, 3)),
#                                G_spread=str(worksheet.cell_value(row, 4)),
#                                yld=str(worksheet.cell_value(row, 5)))
#                 s.save()
#             # SKip #N/A#
#             except:
#                 pass
#
#     return None
#
# def calc_Levenshtein_distance(sec_id):
#     import Levenshtein
#     from operator import itemgetter
#
#     d = {}
#
#     for item in Securities.objects.all().values():
#         string1 = sec_id
#         string2 = str(item['isin'])
#
#         dist = Levenshtein.distance(string1, string2)
#
#         d.update({string2: dist})
#
#     d = sorted(d.items(), key=itemgetter(1))
#     return d[1:11]
#
# def get_vanguard_merge():
#     import xlrd
#
#     workbook = xlrd.open_workbook('./Data/vanguard_merge.xlsx')
#     worksheet = workbook.sheet_by_index(0)
#
#     for row in range(1, worksheet.nrows):
#         try:
#             v = vanguard_merge.objects.get(isin=str(worksheet.cell_value(row, 11)), date=str(worksheet.cell_value(row, 26)))
#         # if there isn’t, create a new entry in the Securities table with the appropriate data
#         except:
#             try:
#                 v = vanguard_merge(bclass3=str(worksheet.cell_value(row,0)),
#                                    country=str(worksheet.cell_value(row,1)),
#                                    bid_spread=float(worksheet.cell_value(row,2)),
#                                    cur_yld=float(worksheet.cell_value(row,3)),
#                                    g_spd=float(worksheet.cell_value(row,4)),
#                                    years_to_mat=float(worksheet.cell_value(row,5)),
#                                    OAS=float(worksheet.cell_value(row,6)),
#                                    OAD=float(worksheet.cell_value(row,7)),
#                                    amt_out=float(worksheet.cell_value(row,8)),
#                                    cpn=float(worksheet.cell_value(row,9)),
#                                    excess_rtn=float(worksheet.cell_value(row,10)),
#                                    ISIN=str(worksheet.cell_value(row,11)),
#                                    ticker=str(worksheet.cell_value(row,12)),
#                                    mty=str(worksheet.cell_value(row,13)),
#                                    iss_dt=str(worksheet.cell_value(row,14)),
#                                    px_close=float(worksheet.cell_value(row,15)),
#                                    KRD_6M=float(worksheet.cell_value(row,16)),
#                                    KRD_2Y=float(worksheet.cell_value(row,17)),
#                                    KRD_5Y=float(worksheet.cell_value(row,18)),
#                                    KRD_10Y=float(worksheet.cell_value(row,19)),
#                                    KRD_20Y=float(worksheet.cell_value(row,20)),
#                                    KRD_30Y=float(worksheet.cell_value(row,21)),
#                                    sp_rating_num=float(worksheet.cell_value(row,22)),
#                                    accrued_int=float(worksheet.cell_value(row,23)),
#                                    yield_to_mat=float(worksheet.cell_value(row,24)),
#                                    class_detail_code=str(worksheet.cell_value(row,25)),
#                                    date=str(worksheet.cell_value(row,26)))
#                 v.save()
#             # SKip #N/A#
#             except:
#                 pass
#
#     return None

