# -*- coding:utf8 -*-

import os

import gspread
import gspread_pandas
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials


class Google_Sheet_Interface(object):
    custom_id_spread_mapping = {}
    credentials_json = ''  # Example: '/path/to/googlesheet-credentials.json'
    scopes = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]

    def __init__(self):
        return

    @property
    def _gspread_client(self):
        return gspread.authorize(
            ServiceAccountCredentials.from_json_keyfile_name(
                self.credentials_json,
                scopes=self.scopes
            )
        )

    @property
    def _gspread_pandas_config(self):
        return gspread_pandas.conf.get_config(
            conf_dir=os.path.dirname(self.credentials_json),
            file_name=os.path.basename(self.credentials_json)
        )

    @classmethod
    def get_spread_id(cls, custom_id):
        return cls.custom_id_spread_mapping[custom_id]

    def _get_gspread_pandas_Spread(self,
                                   custom_id,
                                   ):
        return gspread_pandas.Spread(
            'reco',
            spread=self.get_spread_id(custom_id),
            config=self._gspread_pandas_config,
            create_spread=False,
            create_sheet=False
        )

    def sheet_to_csv(self, spread_id, csv_path):
        raise NotImplementedError

    def csv_to_sheet(self, csv_path, sheet_name):
        raise NotImplementedError

    def sheet_to_df(self,
                    sheet_name,
                    custom_id='-1'):
        if custom_id == '-1':
            custom_id = sheet_name
        return self._get_gspread_pandas_Spread(custom_id).sheet_to_df(index=None,
                                                                      sheet=sheet_name)

    def df_to_sheet(self,
                    df,
                    sheet_name,
                    custom_id='-1'):
        if custom_id == '-1':
            custom_id = sheet_name
        gspread_pandas_spread = self._get_gspread_pandas_Spread(custom_id)
        gspread_pandas_spread.open_sheet(sheet_name,
                                         create=True)
        gspread_pandas_spread.df_to_sheet(df,
                                          index=False,
                                          replace=True)


class Reco_Google_Sheet_Interface(Google_Sheet_Interface):
    custom_id_spread_mapping = {}
    credentials_json = '{dir_path}/conf/googlesheet-credentials.json'.format(
        dir_path=os.path.dirname(os.path.realpath(_file_))
    )
    scopes = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]


class Bundle_Rules_Google_Sheet_Interface(Reco_Google_Sheet_Interface):
    custom_id_spread_mapping = {}
    bundle_rules_column_names = (

    )

    @classmethod
    def get_spread_id_from_custom_id(cls, custom_id):
        return cls.custom_id_spread_mapping[custom_id]

    def custom_id_to_df_bundle_rules(self, custom_id):
        spread = self._gspread_client.open_by_key(
            self.get_spread_id_from_custom_id(custom_id)
        )
        bundles_rules_worksheet = spread.worksheet("Rules")
        first_row = bundles_rules_worksheet.find('BUNDLE RULE NAME').row
        df_bundle_rules = pd.DataFrame({
            column_name: bundles_rules_worksheet.col_values(
                bundles_rules_worksheet.find(column_name).col
            )[first_row:]
            for column_name in self.bundle_rules_column_names
        })
        return df_bundle_rules
