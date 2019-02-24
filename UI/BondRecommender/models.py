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