# DrivenData OpenCities competition  
[Ссылка на соревнование](https://drivendata.org/competitions/60/building-segmentation-disaster-resilience/)  

Ключевой момент этого решения - нарезание на патчи, обучение и инференс работают конфиг файлах. По умолчанию берутся параметры из `configs/base.yaml` но их можно перезаписывать передавая скриптам `-c {path to config file}`. Под капотом это все работает на библиотеке `configargparse` которая является надстройкой над `argparse`. Это позволяет при желании перезаписывать любые аргументы прямо из командой строки без необходимости менять конфиг.  

Допустим конфиг лежит в `configs/my_config.yaml`, тогда все запускается примерно так:  
* `python3 src/slicer.py -c configs/my_config.yaml` - нарежет патчи для тренировки и раздели их на трейн и валидацию  
* `python3 train.py -c configs/my_config.yaml` - обучит модельку и сохранит её куда-то в логи  
* `python3 train.py -c logs/{name of your run}/config.yaml` - запустит инференс на папке с обученной моделью  