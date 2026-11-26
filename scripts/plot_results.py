"""Compatibility entry for aggregate result plots; manuscript artwork is separate."""
from pathlib import Path
import sys
from generate_tables import main
if __name__=='__main__':
 root=Path(__file__).resolve().parents[1]
 sys.argv=[sys.argv[0],'--input',str(root/'results/main_seed_summary.csv'),'--out',str(root/'verification/rendered_figures')]
 main()
