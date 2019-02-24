from django.shortcuts import render
from BondRecommender.models import Securities

# Create your views here.
def home(request):
    context = dict()
    return render(request, 'home.html', context)

def results(request):
    get_securities()

    context = dict()
    sec_id = str(request.GET['sec_id'])
    context['sec_id'] = sec_id
    context['sec_id_error'] = 0

    try:
        Securities.objects.get(isin=sec_id)
        context['similar_bonds'] = calc_Levenshtein_distance(sec_id)
        return render(request, 'results.html', context)
    except:
        context['sec_id_error'] = 1
        return render(request, 'home.html', context)

def get_securities():
    import xlrd

    workbook = xlrd.open_workbook('./Data/SampleData.xlsx')
    worksheet = workbook.sheet_by_index(0)

    for row in range(1, worksheet.nrows):
        #for col in range(worksheet.ncols):
        try:
            s = Securities.objects.get(isin=str(worksheet.cell_value(row, 0)))
        # if there isn’t, create a new entry in the Securities table with the appropriate data
        except:
            try:
                s = Securities(isin=str(worksheet.cell_value(row, 0)),
                               YAS_price=float(worksheet.cell_value(row, 1)),
                               OAS_spread=str(worksheet.cell_value(row, 2)),
                               modified_duration=str(worksheet.cell_value(row, 3)),
                               G_spread=str(worksheet.cell_value(row, 4)),
                               yld=str(worksheet.cell_value(row, 5)))
                s.save()
            # SKip #N/A#
            except:
                pass

    return None

def calc_Levenshtein_distance(sec_id):
    import Levenshtein
    from operator import itemgetter

    d = {}

    for item in Securities.objects.all().values():
        string1 = sec_id
        string2 = str(item['isin'])

        dist = Levenshtein.distance(string1, string2)

        d.update({string2: dist})

    d = sorted(d.items(), key=itemgetter(1))
    return d[1:11]