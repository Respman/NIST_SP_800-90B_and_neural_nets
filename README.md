## NIST SP 800-90B и нейронные сети

# Как запустить сеть?

В папке с конфигом выполняется команда

`python3 ..\..\Neural_nets\neural_network.py .\config_FNN.json`

В конфиге сожержатся все необходимые параметры для конструирования сети.

# Как сгенерировать гамму?

Гамму можно сгенерировать необходимым для исследования источником: 
1. `gamma_sources\LFSR\gen_lfsr_gamma.py` (гамма, снятая с ЛРС), 
2. `gamma_sources\Markov\gen_markov_gamma.py` (гамма, сгенерированная марковским источником),
3. `gamma_sources\Period\gen_period_gamma.py` (гамма, снятая с периодического источника).

Далее корректируется выбранный конфиг и запускается обучение нужной сети.