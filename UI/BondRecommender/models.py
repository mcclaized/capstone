from django.db import models

# Create your models here.
class Securities(models.Model):
    isin = models.CharField(max_length=16)
    YAS_price = models.FloatField()
    OAS_spread = models.CharField(max_length=20)
    modified_duration = models.CharField(max_length=20)
    G_spread = models.CharField(max_length=20)
    yld = models.CharField(max_length=20)
    def __repr__(self):
        return str(self.isin) + ' ' + str(self.YAS_price) + ' ' + str(self.OAS_spread) + \
               ' ' + str(self.modified_duration) + ' ' + str(self.G_spread) + ' ' + str(self.yld)

class vanguard_merge(models.Model):
    bclass3 = models.CharField(max_length=16)
    country = models.CharField(max_length=16)
    bid_spread = models.FloatField()
    cur_yld = models.FloatField()
    g_spd = models.FloatField()
    years_to_mat = models.FloatField()
    OAS = models.FloatField()
    OAD = models.FloatField()
    amt_out = models.FloatField()
    cpn = models.FloatField()
    excess_rtn = models.FloatField()
    ISIN = models.CharField(max_length=16)
    ticker = models.CharField(max_length=16)
    mty = models.CharField(max_length=16)
    iss_dt = models.CharField(max_length=16)
    px_close = models.FloatField()
    KRD_6M = models.FloatField()
    KRD_2Y = models.FloatField()
    KRD_5Y = models.FloatField()
    KRD_10Y = models.FloatField()
    KRD_20Y = models.FloatField()
    KRD_30Y = models.FloatField()
    sp_rating_num = models.FloatField()
    accrued_int = models.FloatField()
    yield_to_mat = models.FloatField()
    class_detail_code = models.CharField(max_length=16)
    date = models.CharField(max_length=16)
    def __repr__(self):
        return str(self.ISIN) + ' ' + str(self.bclass3) + ' ' + str(self.ticker) + \
               ' ' + str(self.date)