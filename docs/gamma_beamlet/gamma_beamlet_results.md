# Beamlet-scale gamma index -- results

32 beamlets, NVIDIA A40, python 3.10.19 / pymedphys 0.41.0 / torch 2.8.0+cu128.

## Accuracy against pymedphys

| path | criterion | rung | backend | beamlets | max |d| (pp) | mean |d| (pp) | worst beamlet |
|---|---|---|---|---|---|---|---|
| array | 1%/1mm/10% | 2 | torch cpu float64 | 8 | 0.000000 | 0.000000 | 3de01567 |
| array | 1%/1mm/10% | 3 | torch cuda:1 float64 | 32 | 0.000000 | 0.000000 | ffe59e68 |
| array | 1%/1mm/10% | 4 | torch cuda:1 float32 | 32 | 0.047192 | 0.012602 | 510113c3 |
| array | 2%/2mm/10% | 2 | torch cpu float64 | 8 | 0.000000 | 0.000000 | 3de01567 |
| array | 2%/2mm/10% | 3 | torch cuda:1 float64 | 32 | 0.000000 | 0.000000 | ffe59e68 |
| array | 2%/2mm/10% | 4 | torch cuda:1 float32 | 32 | 0.028769 | 0.001811 | f56ab9b1 |
| array | 3%/3mm/10% | 2 | torch cpu float64 | 8 | 0.000000 | 0.000000 | 3de01567 |
| array | 3%/3mm/10% | 3 | torch cuda:1 float64 | 32 | 0.000000 | 0.000000 | ffe59e68 |
| array | 3%/3mm/10% | 4 | torch cuda:1 float32 | 32 | 0.025056 | 0.001190 | 9247005d |
| tensor | 1%/1mm/10% | 3 | torch cuda:1 float64 | 32 | 0.000000 | 0.000000 | ffe59e68 |
| tensor | 1%/1mm/10% | 4 | torch cuda:1 float32 | 32 | 0.047192 | 0.012602 | 510113c3 |
| tensor | 2%/2mm/10% | 3 | torch cuda:1 float64 | 32 | 0.000000 | 0.000000 | ffe59e68 |
| tensor | 2%/2mm/10% | 4 | torch cuda:1 float32 | 32 | 0.028769 | 0.001811 | f56ab9b1 |
| tensor | 3%/3mm/10% | 3 | torch cuda:1 float64 | 32 | 0.000000 | 0.000000 | ffe59e68 |
| tensor | 3%/3mm/10% | 4 | torch cuda:1 float32 | 32 | 0.025056 | 0.001190 | 9247005d |

## Time per beamlet

| path | criterion | rung | backend | median (s) | min (s) | max (s) | speed-up | beamlets/s |
|---|---|---|---|---|---|---|---|---|
| array | 1%/1mm/10% | 1 | pymedphys cpu | 0.6469 | 0.1454 | 6.2256 | 1.0x | 1.5 |
| array | 1%/1mm/10% | 2 | torch cpu float64 | 0.6357 | 0.3000 | 1.0736 | 1.0x | 1.6 |
| array | 1%/1mm/10% | 3 | torch cuda:1 float64 | 0.0918 | 0.0563 | 0.2821 | 7.0x | 10.9 |
| array | 1%/1mm/10% | 4 | torch cuda:1 float32 | 0.0931 | 0.0579 | 0.2247 | 7.0x | 10.7 |
| array | 2%/2mm/10% | 1 | pymedphys cpu | 0.2382 | 0.0854 | 3.8505 | 1.0x | 4.2 |
| array | 2%/2mm/10% | 2 | torch cpu float64 | 0.3264 | 0.1544 | 0.6395 | 0.7x | 3.1 |
| array | 2%/2mm/10% | 3 | torch cuda:1 float64 | 0.0585 | 0.0527 | 0.1932 | 4.1x | 17.1 |
| array | 2%/2mm/10% | 4 | torch cuda:1 float32 | 0.0575 | 0.0526 | 0.1385 | 4.1x | 17.4 |
| array | 3%/3mm/10% | 1 | pymedphys cpu | 0.1606 | 0.0748 | 2.3135 | 1.0x | 6.2 |
| array | 3%/3mm/10% | 2 | torch cpu float64 | 0.1756 | 0.0864 | 0.4482 | 0.9x | 5.7 |
| array | 3%/3mm/10% | 3 | torch cuda:1 float64 | 0.0742 | 0.0442 | 0.1519 | 2.2x | 13.5 |
| array | 3%/3mm/10% | 4 | torch cuda:1 float32 | 0.0789 | 0.0469 | 0.1249 | 2.0x | 12.7 |
| tensor | 1%/1mm/10% | 1 | pymedphys cpu | 0.6835 | 0.1335 | 6.2618 | 1.0x | 1.5 |
| tensor | 1%/1mm/10% | 3 | torch cuda:1 float64 | 0.0896 | 0.0553 | 0.2938 | 7.6x | 11.2 |
| tensor | 1%/1mm/10% | 4 | torch cuda:1 float32 | 0.0896 | 0.0569 | 0.2224 | 7.6x | 11.2 |
| tensor | 2%/2mm/10% | 1 | pymedphys cpu | 0.2513 | 0.0959 | 3.6162 | 1.0x | 4.0 |
| tensor | 2%/2mm/10% | 3 | torch cuda:1 float64 | 0.0540 | 0.0507 | 0.1846 | 4.7x | 18.5 |
| tensor | 2%/2mm/10% | 4 | torch cuda:1 float32 | 0.0545 | 0.0507 | 0.1378 | 4.6x | 18.3 |
| tensor | 3%/3mm/10% | 1 | pymedphys cpu | 0.1608 | 0.0786 | 2.3318 | 1.0x | 6.2 |
| tensor | 3%/3mm/10% | 3 | torch cuda:1 float64 | 0.0734 | 0.0427 | 0.1520 | 2.2x | 13.6 |
| tensor | 3%/3mm/10% | 4 | torch cuda:1 float32 | 0.0778 | 0.0468 | 0.1220 | 2.1x | 12.8 |

## One gamma pass over a validation pool

| path | criterion | rung | backend | pool | pass time (s) |
|---|---|---|---|---|---|
| array | 1%/1mm/10% | 1 | pymedphys cpu | 20 | 12.9 |
| array | 1%/1mm/10% | 1 | pymedphys cpu | 200 | 129.4 |
| array | 1%/1mm/10% | 1 | pymedphys cpu | 2000 | 1293.8 |
| array | 1%/1mm/10% | 2 | torch cpu float64 | 20 | 12.7 |
| array | 1%/1mm/10% | 2 | torch cpu float64 | 200 | 127.1 |
| array | 1%/1mm/10% | 2 | torch cpu float64 | 2000 | 1271.4 |
| array | 1%/1mm/10% | 3 | torch cuda:1 float64 | 20 | 1.8 |
| array | 1%/1mm/10% | 3 | torch cuda:1 float64 | 200 | 18.4 |
| array | 1%/1mm/10% | 3 | torch cuda:1 float64 | 2000 | 183.6 |
| array | 1%/1mm/10% | 4 | torch cuda:1 float32 | 20 | 1.9 |
| array | 1%/1mm/10% | 4 | torch cuda:1 float32 | 200 | 18.6 |
| array | 1%/1mm/10% | 4 | torch cuda:1 float32 | 2000 | 186.2 |
| array | 2%/2mm/10% | 1 | pymedphys cpu | 20 | 4.8 |
| array | 2%/2mm/10% | 1 | pymedphys cpu | 200 | 47.6 |
| array | 2%/2mm/10% | 1 | pymedphys cpu | 2000 | 476.4 |
| array | 2%/2mm/10% | 2 | torch cpu float64 | 20 | 6.5 |
| array | 2%/2mm/10% | 2 | torch cpu float64 | 200 | 65.3 |
| array | 2%/2mm/10% | 2 | torch cpu float64 | 2000 | 652.9 |
| array | 2%/2mm/10% | 3 | torch cuda:1 float64 | 20 | 1.2 |
| array | 2%/2mm/10% | 3 | torch cuda:1 float64 | 200 | 11.7 |
| array | 2%/2mm/10% | 3 | torch cuda:1 float64 | 2000 | 117.0 |
| array | 2%/2mm/10% | 4 | torch cuda:1 float32 | 20 | 1.1 |
| array | 2%/2mm/10% | 4 | torch cuda:1 float32 | 200 | 11.5 |
| array | 2%/2mm/10% | 4 | torch cuda:1 float32 | 2000 | 115.0 |
| array | 3%/3mm/10% | 1 | pymedphys cpu | 20 | 3.2 |
| array | 3%/3mm/10% | 1 | pymedphys cpu | 200 | 32.1 |
| array | 3%/3mm/10% | 1 | pymedphys cpu | 2000 | 321.3 |
| array | 3%/3mm/10% | 2 | torch cpu float64 | 20 | 3.5 |
| array | 3%/3mm/10% | 2 | torch cpu float64 | 200 | 35.1 |
| array | 3%/3mm/10% | 2 | torch cpu float64 | 2000 | 351.3 |
| array | 3%/3mm/10% | 3 | torch cuda:1 float64 | 20 | 1.5 |
| array | 3%/3mm/10% | 3 | torch cuda:1 float64 | 200 | 14.8 |
| array | 3%/3mm/10% | 3 | torch cuda:1 float64 | 2000 | 148.5 |
| array | 3%/3mm/10% | 4 | torch cuda:1 float32 | 20 | 1.6 |
| array | 3%/3mm/10% | 4 | torch cuda:1 float32 | 200 | 15.8 |
| array | 3%/3mm/10% | 4 | torch cuda:1 float32 | 2000 | 157.7 |
| tensor | 1%/1mm/10% | 1 | pymedphys cpu | 20 | 13.7 |
| tensor | 1%/1mm/10% | 1 | pymedphys cpu | 200 | 136.7 |
| tensor | 1%/1mm/10% | 1 | pymedphys cpu | 2000 | 1367.0 |
| tensor | 1%/1mm/10% | 3 | torch cuda:1 float64 | 20 | 1.8 |
| tensor | 1%/1mm/10% | 3 | torch cuda:1 float64 | 200 | 17.9 |
| tensor | 1%/1mm/10% | 3 | torch cuda:1 float64 | 2000 | 179.1 |
| tensor | 1%/1mm/10% | 4 | torch cuda:1 float32 | 20 | 1.8 |
| tensor | 1%/1mm/10% | 4 | torch cuda:1 float32 | 200 | 17.9 |
| tensor | 1%/1mm/10% | 4 | torch cuda:1 float32 | 2000 | 179.2 |
| tensor | 2%/2mm/10% | 1 | pymedphys cpu | 20 | 5.0 |
| tensor | 2%/2mm/10% | 1 | pymedphys cpu | 200 | 50.3 |
| tensor | 2%/2mm/10% | 1 | pymedphys cpu | 2000 | 502.5 |
| tensor | 2%/2mm/10% | 3 | torch cuda:1 float64 | 20 | 1.1 |
| tensor | 2%/2mm/10% | 3 | torch cuda:1 float64 | 200 | 10.8 |
| tensor | 2%/2mm/10% | 3 | torch cuda:1 float64 | 2000 | 108.0 |
| tensor | 2%/2mm/10% | 4 | torch cuda:1 float32 | 20 | 1.1 |
| tensor | 2%/2mm/10% | 4 | torch cuda:1 float32 | 200 | 10.9 |
| tensor | 2%/2mm/10% | 4 | torch cuda:1 float32 | 2000 | 109.1 |
| tensor | 3%/3mm/10% | 1 | pymedphys cpu | 20 | 3.2 |
| tensor | 3%/3mm/10% | 1 | pymedphys cpu | 200 | 32.2 |
| tensor | 3%/3mm/10% | 1 | pymedphys cpu | 2000 | 321.7 |
| tensor | 3%/3mm/10% | 3 | torch cuda:1 float64 | 20 | 1.5 |
| tensor | 3%/3mm/10% | 3 | torch cuda:1 float64 | 200 | 14.7 |
| tensor | 3%/3mm/10% | 3 | torch cuda:1 float64 | 2000 | 146.7 |
| tensor | 3%/3mm/10% | 4 | torch cuda:1 float32 | 20 | 1.6 |
| tensor | 3%/3mm/10% | 4 | torch cuda:1 float32 | 200 | 15.6 |
| tensor | 3%/3mm/10% | 4 | torch cuda:1 float32 | 2000 | 155.7 |

