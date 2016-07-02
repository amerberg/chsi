import pandas as pd
import numpy as np
import os

class CHSIDataHandler:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._cache = {}
        
    def csv_path(self, name):
        filename = self.filename(name)
        return os.path.join(self.data_dir, filename)
        
    def filename(self, name):
        base = name.replace('_', '').upper()
        #Account for typo in filename
        if base == 'VULNERABLEPOPSANDENVHEALTH':
            base = 'VUNERABLEPOPSANDENVHEALTH' 
        return base + '.csv'
        
    def csv_parameters(self, name):
        county_index = ['State_FIPS_Code', 'County_FIPS_Code']
        na_values = ['-9999', '-2222', '-2222.2', '-2', '-1111', '-1111.1', '-1', '-9998.9']
        parameters = {
            'DATA_ELEMENT_DESCRIPTION' : {'index_col' : ['PAGE_NAME', 'COLUMN_NAME'], 'na_values' : na_values},
            'DEFINED_DATA_VALUE' : {'index_col' : ['Data_Value'], 'na_values' : na_values},
            'HEALTHY_PEOPLE_2010' : {},
            'DEMOGRAPHICS' : {'index_col' : county_index, 'na_values' : na_values},
            'LEADING_CAUSES_OF_DEATH' : {'index_col' : county_index, 'na_values' : na_values},
            'SUMMARY_MEASURES_OF_HEALTH' : {'index_col' : county_index, 'na_values' : na_values},
            'MEASURES_OF_BIRTH_AND_DEATH' : {'index_col' : county_index, 'na_values' : na_values},
            'RELATIVE_HEALTH_IMPORTANCE' : {'index_col' : county_index, 'na_values' : na_values},
            'VULNERABLE_POPS_AND_ENV_HEALTH' : {'index_col' : county_index, 'na_values' : na_values},
            'PREVENTIVE_SERVICES_USE' : {'index_col' : county_index, 'na_values' : na_values},
            'RISK_FACTORS_AND_ACCESS_TO_CARE' : {'index_col' : county_index, 'na_values' : na_values}
        }
        
        return parameters[name]
        
    def _load_csv(self, name):
        return pd.read_csv(self.csv_path(name), **self.csv_parameters(name))
        
    def get_page(self, name):
        try:
            return self._cache[name]
        except KeyError:
            self._cache[name] = self._load_csv(name)
            self._cache[name].sort_index(inplace=True)
            return self._cache[name]
            
    def data_descriptions(self, page=None):
        descriptions = self.get_page('DATA_ELEMENT_DESCRIPTION')
        if page is None:
            return descriptions
        else:
            return descriptions.loc[(self.page_name(page), slice(None)),:]
        
    def page_name(self, name):
        return name.title().replace('_', '')
    
    def data_element(self, name, page=None):
        if page is not None:
            return dict(self.data_descriptions().loc[self.page_name(page), name])
        else:
            return dict(self.data_descriptions().loc[(slice(None), name), :].iloc[0])
            
    def elements_by_type(self, page, dtype):
        page_elements = self.data_descriptions(page)
        return page_elements[page_elements.DATA_TYPE == dtype]
        
    def county_data_pages(self):
        return ['DEMOGRAPHICS', 'LEADING_CAUSES_OF_DEATH', 
                'SUMMARY_MEASURES_OF_HEALTH',
                'MEASURES_OF_BIRTH_AND_DEATH', 'RELATIVE_HEALTH_IMPORTANCE',
                'RISK_FACTORS_AND_ACCESS_TO_CARE', 'PREVENTIVE_SERVICES_USE',
                'VULNERABLE_POPS_AND_ENV_HEALTH']
                
    def all_county_data(self):
        try:
            return self._all_county_data
        except AttributeError:
            pages = [self.get_page(page) for page in self.county_data_pages()]
            dup_names = ['CHSI_State_Name', 'CHSI_County_Name', 
                         'CHSI_State_Abbr', 'Strata_ID_Number']
                     
            common_cols = pages[0].loc[:, dup_names]
            pieces = [common_cols] + [page.drop(dup_names, axis=1) 
                                      for page in pages]
            
            self._all_county_data = pd.concat(pieces, axis=1)
            return self._all_county_data
            
    def county_data_with_health_status(self):
        county_data = self.all_county_data()
        return county_data.loc[county_data.Health_Status.notnull(),:]
        
    def county_data_good_columns(self, threshold):
        all_cols = self.county_data_with_health_status()
        return all_cols.loc[:,(all_cols.isnull()).mean(axis=0)<threshold]
        
    def drop_columns(self, data):
        drop = [name for name in data.columns if self._non_county_col(name)]
        drop += ['Strata_ID_Number']
        data.drop(drop, axis=1, inplace=True)
    
    def _non_county_col(self, name):
        prefix = ['CI', 'Min', 'Max', 'US']
        suffix = ['Exp']
        return name.split('_')[0] in prefix or name.split('_')[-1] in suffix
    
    def normalize_by_population(self, data):
        col_names = ['Uninsured', 'Disabled_Medicare', 'Elderly_Medicare',
                     'Unemployed', 'Ecol_Rpt', 'Salm_Rpt', 'Shig_Rpt',
                     'CRS_Rpt', 'FluB_Rpt', 'HepA_Rpt', 'HepB_Rpt', 
                     'Pert_Rpt', 'Syphilis_Rpt', 'Meas_Rpt', 
                     'Total_Births', 'Total_Deaths', 'Recent_Drug_Use',
                     'Sev_Work_Disabled', 'Major_Depression']
        for name in col_names:
            try:
                data[name] = 100 * data[name] / data['Population_Size']
            except KeyError:
                pass
            
    def normalize_by_area(self, data):
        col_names = ['Toxic_Chem'] 
        area = data['Population_Size'] / data['Population_Density']
        for name in col_names:
            try:
                data[name] /= area
            except KeyError:
                pass
            
    def normalize_by_years(self, data):
        #TODO: also deal with length of time intervals
        years = self.mbd().MOBD_Time_Span.str.split('-')
        span = years.str.get(1).astype(int)-years.str.get(0).astype(int)+1
        
        col_names = ['Ecol_Rpt', 'Salm_Rpt', 'Shig_Rpt',
                     'CRS_Rpt', 'FluB_Rpt', 'HepA_Rpt', 'HepB_Rpt', 
                     'Pert_Rpt', 'Syphilis_Rpt', 'Meas_Rpt', 
                     'Total_Births', 'Total_Deaths']
                     
        for name in col_names:
            try:
                data[name] /= span
            except KeyError:
                pass
        
    def impute_missing(self, data):
        defaults = data.mean()
        data.fillna(value = defaults[data.columns], inplace=True)
        
    def fix_indicators(self, data):
        #Make all indicator columns +/-1
        for col_name in data.columns:
            if col_name.endswith('_Ind'):
                #This throws out the peer component of the RHI indicators
                data[col_name] = 2*(data[col_name] % 2) - 1                    
                
    def prepared_data(self, threshold):
        data = self.county_data_good_columns(threshold).copy()
        self.drop_columns(data)
        self.fix_indicators(data)
        self.normalize_by_population(data)
        self.normalize_by_area(data)
        self.normalize_by_years(data)
        #TODO: should probably drop Broomfield, CO
        self.impute_missing(data)
        
        return data
        
    def training_data(self, threshold):
        all_data = self.prepared_data(threshold)
        X = all_data.select_dtypes(include=[np.number]).drop(['Health_Status'], axis=1)
        Y = all_data.Health_Status
        return X,Y
        
    def demographics(self):
        return self.get_page('DEMOGRAPHICS')
        
    def lcd(self):
        return self.get_page('LEADING_CAUSES_OF_DEATH')
        
    def smh(self):
        return self.get_page('SUMMARY_MEASURES_OF_HEALTH')
        
    def mbd(self):
        return self.get_page('MEASURES_OF_BIRTH_AND_DEATH')
        
    def rhi(self):
        return self.get_page('RELATIVE_HEALTH_IMPORTANCE')
        
    def vpeh(self):
        return self.get_page('VULNERABLE_POPS_AND_ENVIRONMENTAL_HEALTH')
        
    def rfac(self):
        return self.get_page('RISK_FACTORS_AND_ACCESS_TO_CARE')