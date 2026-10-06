#!/usr/bin/env bash
# Executa todos os experimentos em sequência e mostra a comparação.
set -e
for cfg in configs/*.yaml; do
  echo ">>> Executando $cfg"
  python -m src.train --config "$cfg"
done
python -m src.compare
