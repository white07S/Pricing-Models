from models.binomial.binomial import euro_option

option_eu = euro_option(217.58, 215, 0.05, 0.1, 40, {'tk': 'AAPL', 'is_calc': True, 'start': '2021-08-18',
                                                     'end': '2022-08-18', 'eu_option':False})
option_eu2 = euro_option(217.58, 215, 0.05, 0.1, 40, {'tk': 'AAPL', 'is_calc': True, 'use_garch': True, 'start': '2021-08-18',
                                                     'end': '2022-08-18', 'eu_option':False})

print(option_eu.price())
print(option_eu2.price())