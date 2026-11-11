"""Build a reproducible 10,000-row HR attrition workbook from the public HR_comma_sep.csv.

Public source data is not bundled. Download it separately and comply with its license.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

HEADER = [
    'satisfaction_level', 'last_evaluation', 'number_project',
    'average_montly_hours', 'time_spend_company', 'Work_accident',
    'promotion_last_5years', 'Department', 'salary', 'Attrition'
]

class Xorshift32:
    """Replicate the original JavaScript 32-bit seed and Fisher-Yates sampling."""
    def __init__(self, seed: int = 20261111):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        v = self.state
        v ^= (v << 13) & 0xFFFFFFFF
        v ^= v >> 17
        v ^= (v << 5) & 0xFFFFFFFF
        self.state = v & 0xFFFFFFFF
        return (self.state + 0.5) / 4294967296

    def shuffle(self, items: list) -> list:
        for i in range(len(items)-1,0,-1):
            j = int(self.random()*(i+1))
            items[i],items[j] = items[j],items[i]
        return items


def build(source: Path, output: Path, seed=20261111, target_rows=10000):
    with source.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    head = rows.pop(0)
    expected = ['left', 'last_evaluation', 'number_project', 'average_montly_hours',
                'time_spend_company', 'Work_accident', 'satisfaction_level',
                'promotion_last_5years', 'sales', 'salary']
    if head != expected:
        raise ValueError(f'Unexpected source columns: {head}')

    seen = set()
    unique = []
    for row in rows:
        if len(row) != 10 or row[0] not in ('0','1'):
            raise ValueError(f'Unexpected row: {row}')
        # Deduplicate on all predictor values (not target). Keep first occurrence.
        key = tuple(row[1:])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if len(unique) < target_rows:
        raise ValueError(f'Only {len(unique)} unique rows, need {target_rows}')
    groups = {'0': [], '1': []}
    for row in unique:
        groups[row[0]].append(row)
    rng = Xorshift32(seed)
    rng.shuffle(groups['0'])
    rng.shuffle(groups['1'])
    yes_count = round(target_rows*len(groups['1'])/len(unique))
    no_count = target_rows - yes_count
    selected = rng.shuffle(groups['0'][:no_count] + groups['1'][:yes_count])

    wb = Workbook()
    ws = wb.active
    ws.title = 'Attrition_Data'
    ws.append(HEADER)
    for row in selected:
        a = [float(row[6]), float(row[1]), int(row[2]), int(row[3]),
             int(row[4]), int(row[5]), int(row[7]), row[8], row[9],
             'Yes' if row[0]=='1' else 'No']
        ws.append(a)
    ws.freeze_panes='A2'
    ws.auto_filter.ref=f'A1:J{target_rows+1}'
    ws.row_dimensions[1].height=25
    for col in range(1,11):
        ws.column_dimensions[get_column_letter(col)].width=24 if col!=8 else 19
    for cell in ws[1]:
        cell.fill=PatternFill('solid',fgColor='E7EEF4')
        cell.font=Font(bold=True,color='0A1934')
        cell.alignment=Alignment(vertical='center',wrap_text=True)
    info=wb.create_sheet('Dataset_Info')
    for row in [
        ['DATASET INFORMATION','VALUE'],['Source file',source.name],
        ['Original rows',len(rows)],['Rows after predictor deduplication',len(unique)],
        ['Selected rows',len(selected)],['Attrition Yes',yes_count],
        ['Attrition No',no_count],['Sampling seed',seed],
        ['Source reference','https://github.com/RuchitaGarde/Human-Resources-Analytics'],
        ['Source warning','Educational/public benchmark; not verified real employee records'],
        ['License warning','Source dataset copyright/license must be reviewed before public redistribution'],
    ]:info.append(row)
    info.column_dimensions['A'].width=39
    info.column_dimensions['B'].width=98
    for cell in info[1]:
        cell.fill=PatternFill('solid',fgColor='E7EEF4')
        cell.font=Font(bold=True,color='0A1934')
    output.parent.mkdir(parents=True,exist_ok=True)
    wb.save(output)
    return {'original':len(rows),'unique':len(unique),'selected':len(selected),'left':yes_count,'stayed':no_count}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Rebuild the 10,000-row training workbook.')
    parser.add_argument('--source',type=Path,required=True,help='Path to downloaded HR_comma_sep.csv')
    parser.add_argument('--output',type=Path,default=Path('data/Employee_Attrition_10000_Training_Dataset.xlsx'))
    args=parser.parse_args()
    print(build(args.source,args.output))
