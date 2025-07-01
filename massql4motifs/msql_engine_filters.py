import pandas as pd
from py_expression_eval import Parser
math_parser = Parser()

def _get_mz_tolerance(qualifiers, mz):
    if qualifiers is None:
        return 0.1

    if "qualifierppmtolerance" in qualifiers:
        ppm = qualifiers["qualifierppmtolerance"]["value"]
        mz_tol = abs(ppm * mz / 1000000)
        return mz_tol

    if "qualifiermztolerance" in qualifiers:
        return qualifiers["qualifiermztolerance"]["value"]

    return 0.1

def _get_massdefect_min(qualifiers):
    if qualifiers is None:
        return 0, 1

    if "qualifiermassdefect" in qualifiers:
        return qualifiers["qualifiermassdefect"]["min"], qualifiers["qualifiermassdefect"]["max"]
    
    return 0, 1


def _get_minintensity(qualifier):
    """
    Returns absolute min and relative min

    Args:
        qualifier ([type]): [description]

    Returns:
        [type]: [description]
    """

    min_intensity = 0
    min_percent_intensity = 0
    min_tic_percent_intensity = 0
    

    if qualifier is None:
        min_intensity = 0
        min_percent_intensity = 0

        return min_intensity, min_percent_intensity, min_tic_percent_intensity
    
    if "qualifierintensityvalue" in qualifier:
        min_intensity = float(qualifier["qualifierintensityvalue"]["value"])

    if "qualifierintensitypercent" in qualifier:
        min_percent_intensity = float(qualifier["qualifierintensitypercent"]["value"]) / 100

    if "qualifierintensityticpercent" in qualifier:
        min_tic_percent_intensity = float(qualifier["qualifierintensityticpercent"]["value"]) / 100

    # since the subsequent comparison is a strict greater than, if people set it to 100, then they won't get anything. 
    min_percent_intensity = min(min_percent_intensity, 0.99)

    return min_intensity, min_percent_intensity, min_tic_percent_intensity

def _get_exclusion_flag(qualifiers):
    if qualifiers is None:
        return False

    if "qualifierexcluded" in qualifiers:
        return True

    return False

def _get_otherscan_qualifier(qualifiers):
    if qualifiers is None:
        return None

    if "qualifierotherscan" in qualifiers:
        return qualifiers["qualifierotherscan"]

    return None

def _set_intensity_register(ms_filtered_df, register_dict, condition):
    if "qualifiers" in condition:
        if "qualifierintensityreference" in condition["qualifiers"]:
            qualifier_variable = condition["qualifiers"]["qualifierintensitymatch"]["value"]

            grouped_df = ms_filtered_df.groupby("scan").sum().reset_index()
            for grouped_scan in grouped_df.to_dict(orient="records"):
                # Saving into the register
                key = "scan:{}:variable:{}".format(grouped_scan["scan"], qualifier_variable)
                register_dict[key] = grouped_scan["i"]
    return

def _filter_intensitymatch(ms_filtered_df, register_dict, condition):
    if "qualifiers" in condition:
        if "qualifierintensitymatch" in condition["qualifiers"] and \
            "qualifierintensitytolpercent" in condition["qualifiers"]:
            qualifier_expression = condition["qualifiers"]["qualifierintensitymatch"]["value"]
            qualifier_variable = qualifier_expression[0] #TODO: This assumes the variable is the first character in the expression, likely a bad assumption

            grouped_df = ms_filtered_df.groupby("scan").sum().reset_index()

            filtered_grouped_scans = []
            for grouped_scan in grouped_df.to_dict(orient="records"):
                # Reading from the register
                key = "scan:{}:variable:{}".format(grouped_scan["scan"], qualifier_variable)

                if key in register_dict:
                    register_value = register_dict[key]                    
                    evaluated_new_expression = math_parser.parse(qualifier_expression).evaluate({
                        qualifier_variable : register_value
                    })

                    min_match_intensity, max_match_intensity = _get_intensitymatch_range(condition["qualifiers"], evaluated_new_expression)

                    scan_intensity = grouped_scan["i"]

                    #print(key, scan_intensity, qualifier_expression, min_match_intensity, max_match_intensity, grouped_scan)

                    if scan_intensity > min_match_intensity and \
                        scan_intensity < max_match_intensity:
                        filtered_grouped_scans.append(grouped_scan)
                else:
                    # Its not in the register, which means we don't find it
                    continue
            return pd.DataFrame(filtered_grouped_scans)

    return ms_filtered_df

def _get_intensitymatch_range(qualifiers, match_intensity):
    """
    Matching the intensity range

    Args:
        qualifiers ([type]): [description]
        match_intensity ([type]): [description]

    Returns:
        [type]: [description]
    """

    min_intensity = 0
    max_intensity = 0

    if "qualifierintensitytolpercent" in qualifiers:
        tolerance_percent = qualifiers["qualifierintensitytolpercent"]["value"]
        tolerance_value = float(tolerance_percent) / 100 * match_intensity
        
        min_intensity = match_intensity - tolerance_value
        max_intensity = match_intensity + tolerance_value

    return min_intensity, max_intensity

def _merge_filter_cardinality(condition, ms_df_list):
    if "qualifiers" in condition:
        if "qualifiercardinality" in condition["qualifiers"]:
            min_cardinality = condition["qualifiers"]["qualifiercardinality"]["min"]
            max_cardinality = condition["qualifiers"]["qualifiercardinality"]["max"]

            # Figuring out the scans
            ms_peak_df = pd.concat(ms_df_list)

            enumeration_df = ms_peak_df.groupby(["scan", "mzenumeration"]).first().reset_index()
            enumeration_df = enumeration_df.groupby(["scan"]).count()
            enumeration_df = enumeration_df[enumeration_df["mzenumeration"] >= min_cardinality]
            enumeration_df = enumeration_df[enumeration_df["mzenumeration"] <= max_cardinality]

            scans = list(enumeration_df.index.unique())
            filtered_ms_peak_df = ms_peak_df[ms_peak_df["scan"].isin(scans)]
        else:
            filtered_ms_peak_df = pd.concat(ms_df_list)
    else:
        filtered_ms_peak_df = pd.concat(ms_df_list)

    return filtered_ms_peak_df

def ms2prod_condition(condition, ms1_df, ms2_df, reference_conditions_register):
    """
    Filters the MS1 and MS2 data based upon MS2 peak conditions

    Args:
        condition ([type]): [description]
        ms1_df ([type]): [description]
        ms2_df ([type]): [description]
        reference_conditions_register ([type]): Edits this in place

    Returns:
        ms1_df ([type]): [description]
        ms2_df ([type]): [description]
    """
    exclusion_flag = _get_exclusion_flag(condition.get("qualifiers", None))

    if len(ms2_df) == 0:
        return ms1_df, ms2_df

    ms2_list = []
    for i, mz in enumerate(condition["value"]):
        if mz == "ANY":
            # Checking defect options
            massdefect_min, massdefect_max = _get_massdefect_min(condition.get("qualifiers", None))
            ms2_filtered_df = ms2_df
            ms2_filtered_df["mz_defect"] = ms2_filtered_df["frag_mz"] - ms2_filtered_df["frag_mz"].astype(int)

            min_int, min_intpercent, min_tic_percent_intensity = _get_minintensity(condition.get("qualifiers", None))

            ms2_filtered_df = ms2_filtered_df[
                (ms2_filtered_df["mz_defect"] > massdefect_min) & 
                (ms2_filtered_df["mz_defect"] < massdefect_max) &
                (ms2_filtered_df["frag_intens"] > min_int) 
            ]
        else:
            mz_tol = _get_mz_tolerance(condition.get("qualifiers", None), mz)
            mz_min = mz - mz_tol
            mz_max = mz + mz_tol

            min_int, min_intpercent, min_tic_percent_intensity = _get_minintensity(condition.get("qualifiers", None))

            ms2_filtered_df = ms2_df[
                (ms2_df["frag_mz"] > mz_min) & 
                (ms2_df["frag_mz"] < mz_max) & 
                (ms2_df["frag_intens"] > min_int) 
            ]

        # Setting the intensity match register
        _set_intensity_register(ms2_filtered_df, reference_conditions_register, condition)

        # Applying the intensity match
        ms2_filtered_df = _filter_intensitymatch(ms2_filtered_df, reference_conditions_register, condition).copy()
        ms2_filtered_df["mzenumeration"] = i

        ms2_list.append(ms2_filtered_df)

    if len(ms2_list) == 1:
        ms2_filtered_df = ms2_list[0]
    else:
        ms2_filtered_df = _merge_filter_cardinality(condition, ms2_list)

    # Apply the negation operator
    if exclusion_flag:
        filtered_scans = set(ms2_filtered_df["scan"])
        original_scans = set(ms2_df["scan"])
        negation_scans = original_scans - filtered_scans

        ms2_filtered_df = ms2_df[ms2_df["scan"].isin(negation_scans)]

    if len(ms2_filtered_df) == 0:
       return pd.DataFrame(), pd.DataFrame()
    
    # Filtering the actual data structures
    filtered_scans = set(ms2_filtered_df["scan"])
    ms2_df = ms2_df[ms2_df["scan"].isin(filtered_scans)]

    # Filtering the MS1 data now
    ms1_scans = set(ms2_df["ms1scan"])
    ms1_df = ms1_df[ms1_df["scan"].isin(ms1_scans)]

    return ms1_df, ms2_df

def ms2nl_condition(condition, ms1_df, ms2_df, reference_conditions_register):
    """
    Filters the MS1 and MS2 data based upon MS2 neutral loss conditions

    Args:
        condition ([type]): [description]
        ms1_df ([type]): [description]
        ms2_df ([type]): [description]
        reference_conditions_register ([type]): Edits this in place

    Returns:
        ms1_df ([type]): [description]
        ms2_df ([type]): [description]
    """
    exclusion_flag = _get_exclusion_flag(condition.get("qualifiers", None))

    if len(ms2_df) == 0:
        return ms1_df, ms2_df

    ms2_list = []
    for mz in condition["value"]:
        if mz == "ANY":
            # Checking defect options
            massdefect_min, massdefect_max = _get_massdefect_min(condition.get("qualifiers", None))
            ms2_filtered_df = ms2_df
            ms2_filtered_df["mz_defect"] = ms2_filtered_df["frag_mz"] - ms2_filtered_df["frag_mz"].astype(int)

            min_int, min_intpercent, min_tic_percent_intensity = _get_minintensity(condition.get("qualifiers", None))

            ms2_filtered_df = ms2_filtered_df[
                (ms2_filtered_df["mz_defect"] > massdefect_min) & 
                (ms2_filtered_df["mz_defect"] < massdefect_max) &
                (ms2_filtered_df["frag_intens"] > min_int) 
            ]
        else:
            mz_tol = _get_mz_tolerance(condition.get("qualifiers", None), mz) #TODO: This is incorrect logic if it comes to PPM accuracy
            nl_min = mz - mz_tol
            nl_max = mz + mz_tol

            min_int, min_intpercent, min_tic_percent_intensity = _get_minintensity(condition.get("qualifiers", None))

            ms2_filtered_df = ms2_df[
                (ms2_df["loss_mz"] > nl_min) & 
                (ms2_df["loss_mz"] < nl_max) &
                (ms2_df["loss_intens"] > min_int) 
            ]

        # Setting the intensity match register
        _set_intensity_register(ms2_filtered_df, reference_conditions_register, condition)

        # Applying the intensity match
        ms2_filtered_df = _filter_intensitymatch(ms2_filtered_df, reference_conditions_register, condition)

        ms2_list.append(ms2_filtered_df)

    if len(ms2_list) == 1:
        ms2_filtered_df = ms2_list[0]
    else:
        ms2_filtered_df = _merge_filter_cardinality(condition, ms2_list)

    # Apply the negation operator
    if exclusion_flag:
        filtered_scans = set(ms2_filtered_df["scan"])
        original_scans = set(ms2_df["scan"])
        negation_scans = original_scans - filtered_scans

        ms2_filtered_df = ms2_df[ms2_df["scan"].isin(negation_scans)]

    if len(ms2_filtered_df) == 0:
       return pd.DataFrame(), pd.DataFrame()

    # Filtering the actual data structures
    filtered_scans = set(ms2_filtered_df["scan"])
    ms2_df = ms2_df[ms2_df["scan"].isin(filtered_scans)]

    # Filtering the MS1 data now
    ms1_scans = set(ms2_df["ms1scan"])
    ms1_df = ms1_df[ms1_df["scan"].isin(ms1_scans)]

    return ms1_df, ms2_df

def ms2prec_condition(condition, ms1_df, ms2_df, reference_conditions_register):
    """
    Filters the MS1 and MS2 data based upon MS2 precursor conditions

    Args:
        condition ([type]): [description]
        ms1_df ([type]): [description]
        ms2_df ([type]): [description]
        reference_conditions_register ([type]): Edits this in place

    Returns:
        ms1_df ([type]): [description]
        ms2_df ([type]): [description]
    """
    exclusion_flag = _get_exclusion_flag(condition.get("qualifiers", None))

    if len(ms2_df) == 0:
        return ms1_df, ms2_df

    ms2_list = []
    for mz in condition["value"]:
        if mz == "ANY":
            # Checking defect options
            massdefect_min, massdefect_max = _get_massdefect_min(condition.get("qualifiers", None))
            ms2_filtered_df = ms2_df
            ms2_filtered_df["precmz_defect"] = ms2_filtered_df["precmz"] - ms2_filtered_df["precmz"].astype(int)

            ms2_filtered_df = ms2_filtered_df[
                (ms2_filtered_df["precmz_defect"] > massdefect_min) & 
                (ms2_filtered_df["precmz_defect"] < massdefect_max)
            ]
        else:
            mz_tol = _get_mz_tolerance(condition.get("qualifiers", None), mz)
            mz_min = mz - mz_tol
            mz_max = mz + mz_tol

            ms2_filtered_df = ms2_df[
                (ms2_df["precmz"] > mz_min) & 
                (ms2_df["precmz"] < mz_max)
            ]

        ms2_list.append(ms2_filtered_df)

    if len(ms2_list) == 1:
        ms2_filtered_df = ms2_list[0]
    else:
        ms2_filtered_df = _merge_filter_cardinality(condition, ms2_list)
    
    # Apply the negation operator
    if exclusion_flag:
        filtered_scans = set(ms2_filtered_df["scan"])
        original_scans = set(ms2_df["scan"])
        negation_scans = original_scans - filtered_scans

        ms2_filtered_df = ms2_df[ms2_df["scan"].isin(negation_scans)]

    if len(ms2_filtered_df) == 0:
       return pd.DataFrame(), pd.DataFrame()
    
    # Filtering the actual data structures
    filtered_scans = set(ms2_filtered_df["scan"])
    ms2_df = ms2_df[ms2_df["scan"].isin(filtered_scans)]

    # Filtering the MS1 data now
    if len(ms1_df) > 0:
        ms1_scans = set(ms2_df["ms1scan"])
        ms1_df = ms1_df[ms1_df["scan"].isin(ms1_scans)]

    return ms1_df, ms2_df

def ms1_condition(condition, ms1_df, ms2_df, reference_conditions_register, ms1_original_df, ms2_original_df):
    """
    Filters the MS1 and MS2 data based upon MS1 peak conditions

    Args:
        condition ([type]): [description]
        ms1_df ([type]): [description]
        ms2_df ([type]): [description]
        reference_conditions_register ([type]): Edits this in place
        ms1_original_df: These are the original MS1 data, unfiltered
        ms2_original_df: These are the original MS2 data, unfiltered

    Returns:
        ms1_df ([type]): [description]
        ms2_df ([type]): [description]
    """
    exclusion_flag = _get_exclusion_flag(condition.get("qualifiers", None))

    if len(ms1_df) == 0:
        return ms1_df, ms2_df

    ms1_list = []
    for mz in condition["value"]:
        if mz == "ANY":
            # Checking defect options
            massdefect_min, massdefect_max = _get_massdefect_min(condition.get("qualifiers", None))
            ms1_filtered_df = ms1_df
            ms1_filtered_df["mz_defect"] = ms1_filtered_df["mz"] - ms1_filtered_df["mz"].astype(int)

            min_int, min_intpercent, min_tic_percent_intensity = _get_minintensity(condition.get("qualifiers", None))

            ms1_filtered_df = ms1_filtered_df[
                (ms1_filtered_df["mz_defect"] > massdefect_min) & 
                (ms1_filtered_df["mz_defect"] < massdefect_max) &
                (ms1_filtered_df["i"] > min_int) & 
                (ms1_filtered_df["i_norm"] > min_intpercent) & 
                (ms1_filtered_df["i_tic_norm"] > min_tic_percent_intensity)
            ]
        else:
            # Checking defect options
            massdefect_min, massdefect_max = _get_massdefect_min(condition.get("qualifiers", None))

            mz_tol = _get_mz_tolerance(condition.get("qualifiers", None), mz)
            mz_min = mz - mz_tol
            mz_max = mz + mz_tol

            min_int, min_intpercent, min_tic_percent_intensity = _get_minintensity(condition.get("qualifiers", None))

            otherscan_qualifier = _get_otherscan_qualifier(condition.get("qualifiers", None))

            if otherscan_qualifier is None:
                ms1_filtered_df = ms1_df[
                    (ms1_df["mz"] > mz_min) & 
                    (ms1_df["mz"] < mz_max) & 
                    (ms1_df["i"] > min_int) & 
                    (ms1_df["i_norm"] > min_intpercent) & 
                    (ms1_df["i_tic_norm"] > min_tic_percent_intensity)]

                if massdefect_min > 0 or massdefect_max < 1:
                    ms1_filtered_df["mz_defect"] = ms1_filtered_df["mz"] - ms1_filtered_df["mz"].astype(int)
                    
                    ms1_filtered_df = ms1_filtered_df[
                        (ms1_filtered_df["mz_defect"] > massdefect_min) & 
                        (ms1_filtered_df["mz_defect"] < massdefect_max)
                    ]
            else:
                # Here we actually have an other scan qualifier, so we need the full data. This functionality is super limited
                grouped_df = ms1_df.groupby("scan").first()

                scans_to_keep = []

                for scan, row in grouped_df.iterrows():
                    current_scan_rt = row["rt"]
                    
                    min_original_rt = current_scan_rt - otherscan_qualifier["min"]
                    max_original_rt = current_scan_rt + otherscan_qualifier["max"]

                    print("RT RANGE", min_original_rt, max_original_rt)

                    print("mz_min", mz_min)
                    print("mz_max", mz_max)
                    print("min_int", min_int)
                    print("min_intpercent", min_intpercent)
                    print("min_tic_percent_intensity", min_tic_percent_intensity)

                    ms1_original_filtered_df = ms1_original_df[
                        (ms1_original_df["mz"] > mz_min) & 
                        (ms1_original_df["mz"] < mz_max) & 
                        (ms1_original_df["i"] > min_int) & 
                        (ms1_original_df["i_norm"] > min_intpercent) & 
                        (ms1_original_df["i_tic_norm"] > min_tic_percent_intensity) &
                        (ms1_original_df["rt"] > min_original_rt) &
                        (ms1_original_df["rt"] < max_original_rt)]
                    
                    if len(ms1_original_filtered_df) > 0:
                        # This means, the current scan we're considering is a scan that is valid to keep
                        scans_to_keep.append(scan)
                    
                # Lets filter the ms1_filtered to only the scans we want to keep
                ms1_filtered_df = ms1_df[ms1_df["scan"].isin(scans_to_keep)]
                

        # Setting the intensity match register
        _set_intensity_register(ms1_filtered_df, reference_conditions_register, condition)

        # Applying the intensity match
        ms1_filtered_df = _filter_intensitymatch(ms1_filtered_df, reference_conditions_register, condition)

        ms1_list.append(ms1_filtered_df)
    
    if len(ms1_list) == 1:
        ms1_filtered_df = ms1_list[0]
    else:
        ms1_filtered_df = _merge_filter_cardinality(condition, ms1_list)

    # Apply the negation operator
    if exclusion_flag:
        filtered_scans = set(ms1_filtered_df["scan"])
        original_scans = set(ms1_df["scan"])
        negation_scans = original_scans - filtered_scans

        ms1_filtered_df = ms1_df[ms1_df["scan"].isin(negation_scans)]

    if len(ms1_filtered_df) == 0:
       return pd.DataFrame(), pd.DataFrame()

    # Filtering the actual data structures
    filtered_scans = set(ms1_filtered_df["scan"])
    ms1_df = ms1_df[ms1_df["scan"].isin(filtered_scans)]

    if "ms1scan" in ms2_df:
        ms2_df = ms2_df[ms2_df["ms1scan"].isin(filtered_scans)]

    return ms1_df, ms2_df

    
def ms1_filter(condition, ms1_df):
    """
    Filters the MS1 and MS2 data based upon MS1 peak filters

    Args:
        condition ([type]): [description]
        ms1_df ([type]): [description]

    Returns:
        ms1_df ([type]): [description]
    """

    if len(ms1_df) == 0:
        return ms1_df

    ms1_list = []
    for mz in condition["value"]:
        if mz == "ANY":
            # Checking defect options
            massdefect_min, massdefect_max = _get_massdefect_min(condition.get("qualifiers", None))
            ms1_filtered_df = ms1_df
            ms1_filtered_df["mz_defect"] = ms1_filtered_df["mz"] - ms1_filtered_df["mz"].astype(int)

            min_int, min_intpercent, min_tic_percent_intensity = _get_minintensity(condition.get("qualifiers", None))

            ms1_filtered_df = ms1_filtered_df[
                (ms1_filtered_df["mz_defect"] > massdefect_min) & 
                (ms1_filtered_df["mz_defect"] < massdefect_max) &
                (ms1_filtered_df["i"] > min_int) & 
                (ms1_filtered_df["i_norm"] > min_intpercent) & 
                (ms1_filtered_df["i_tic_norm"] > min_tic_percent_intensity)
            ]
        else:
            # Checking defect options
            massdefect_min, massdefect_max = _get_massdefect_min(condition.get("qualifiers", None))
            
            mz_tol = _get_mz_tolerance(condition.get("qualifiers", None), mz)
            mz_min = mz - mz_tol
            mz_max = mz + mz_tol

            min_int, min_intpercent, min_tic_percent_intensity = _get_minintensity(condition.get("qualifiers", None))
            ms1_filtered_df = ms1_df[
                (ms1_df["mz"] > mz_min) & 
                (ms1_df["mz"] < mz_max) & 
                (ms1_df["i"] > min_int) & 
                (ms1_df["i_norm"] > min_intpercent) & 
                (ms1_df["i_tic_norm"] > min_tic_percent_intensity)]

            if massdefect_min > 0 or massdefect_max < 1:
                ms1_filtered_df["mz_defect"] = ms1_filtered_df["mz"] - ms1_filtered_df["mz"].astype(int)

                ms1_filtered_df = ms1_filtered_df[
                    (ms1_filtered_df["mz_defect"] > massdefect_min) & 
                    (ms1_filtered_df["mz_defect"] < massdefect_max)
                ]

        ms1_list.append(ms1_filtered_df)
    
    if len(ms1_list) == 1:
        ms1_filtered_df = ms1_list[0]
    else:
        ms1_filtered_df = pd.concat(ms1_list)

    if len(ms1_filtered_df) == 0:
       return pd.DataFrame()

    return ms1_filtered_df
