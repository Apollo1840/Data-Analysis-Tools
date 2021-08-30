import pandas as pd
import matplotlib.pyplot as plt

pro_score = {

    # lage
    "next to asian market": 0.6,
    "near asian market": 0.5,
    "next to asian res": 1,
    "near asian res": 0.5,

    "next to congyu": 0.8,
    "next to shangsu": 0.8,
    "near congyu": 0.5,
    "near shangsu": 0.5,

    "next to park": 0.6,
    "near park": 0.4,

    "next to Ubahn": 0.5,
    "next to market": 0.4,

    "nice bacony": 0.7,
    "nice layout": 1,
    "nice mobi": 2,
    "nice küche": 0.5,

    "nice building": 0.5,
    "high value": 0.2,
}

con_score = {
    "no bacony": -1,  # with replacement

    "far to shangsu": -2,
    "far to congyu": -2,
    "possion": -2,

    "unmobil": -0.1,

    "low value": -0.5,
    "far to Ubahn": -0.2,
}


class AppartmentEntity():

    def __init__(self, base_score, id, price, size, pro, con, spec, label=None):
        self._base_score = base_score
        self.id = id
        self.price = price + 80
        self.size = size
        self.pro = pro
        self.con = con
        self.spec = spec
        self.label = label

    @property
    def desire_score(self):
        desire_score = self._base_score
        for condition, effect in pro_score.items():
            if condition in self.pro:
                desire_score += effect
        for condition, effect in con_score.items():
            if condition in self.con:
                desire_score += effect

        desire_score += self._price_sensitive(self.price)
        return desire_score

    def _serialize(self):
        prop = {}
        for key in dir(self):
            if not key.startswith("_"):
                prop.update({key: getattr(self, key)})
        return prop

    @staticmethod
    def _price_sensitive(price):
        if price < 1000:
            return 1
        elif 1000 <= price < 1050:
            return 0.75
        elif 1050 <= price < 1100:
            return 0.5
        elif 1100 <= price < 1150:
            return 0.25
        elif 1150 <= price < 1200:
            return 0
        elif 1200 <= price < 1250:
            return -0.5
        elif 1250 <= price < 1300:
            return -1

        elif 1300 <= price < 1350:
            return -1.5
        elif 1350 <= price < 1380:
            return -2
        elif 1380 <= price <= 1400:
            return -2.5

        elif 1400 < price <= 1425:
            return -3
        elif 1425 < price <= 1450:
            return -3.5
        elif 1450 < price <= 1470:
            return -4
        elif 1470 < price <= 1490:
            return -4.5
        elif 1490 < price <= 1500:
            return -5

        elif 1500 < price <= 1510:
            return -5.5
        elif 1510 < price <= 1520:
            return -6
        elif 1520 < price <= 1530:
            return -6.5
        elif 1530 < price <= 1540:
            return -7
        elif 1540 < price <= 1550:
            return -7.5

        elif 1550 < price <= 1575:
            return -9
        elif 1575 <= price:
            return -10

    @staticmethod
    def _plot_price_sensitive():
        plt.plot(range(500, 1600), [AppartmentEntity._price_sensitive(p) for p in range(500, 1600)])
        plt.grid()
        plt.show()


data = [
    [5, "app", "hochgross", 1345, 38, "", "far to shangsu; possion; low value",
     "next to park; next to market; nice küche; nice bacony; "],
    [4, "ref", "helle22", 1500, 54, "2 Jahre, no internet", "",
     "nice bacony; nice building; next to market; high value"],
    [1.5, "app", "ab9", 1330, 49, "3000 nach, awful bacony", "bad light, unmobil",
     "nice layout; next to Ubahn; next to park, next to market"],
    [4, "app", "schwabingT", 1247, 40, "2 Jahre", "far to congyu, far to Ubahn", "near shangsu, next to market"],
    [4, "app", "helle&r", 1390, 56, "200 nach", "no lift with 1 floor, old building",
     "next to Ubahn; next to market; near park, nice layout"],
    [5, "app", "ruhige2.5", 1299, 82, "", "far to shangsu, no bacony with workroom",
     "nice layout, next to market, next to Ubahn, high value"],
    [5, "", "nette2", 1230, 63, "", "far to shangsu", "near congyu, next to market"]
]

gepflegte2 = AppartmentEntity(
    base_score=4.8,
    id="gepflegte2",
    price=1310,
    size=53,
    pro="next to Ubahn; nice layout, next to park; next to congyu; ",
    con="no bacony with EG",
    spec="1500 nach; 2 Jahre ",
    label="app"
)

leopold = AppartmentEntity(
    base_score=4.8,
    id="leopold",
    price=1430,
    size=55,
    pro="near park; nice layout; near shangsu; next to market; nice mobi, nice küche",
    con="old building",
    spec="chinese, large sofa, small bedroom, easy gabage",
    label="app"
)

mobschon = AppartmentEntity(
    base_score=5,
    id="mobschon",
    price=1300,
    size=39,
    pro="near park; nice layout, next to market; ",
    con="old building",
    spec="kvr",
    label="app"
)


