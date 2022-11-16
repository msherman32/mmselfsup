from climate_tutorial.utils.datetime import Year, Days, Hours
from climate_tutorial.data import DataModule

data_module = DataModule(
    dataset = "ERA5",
    task = "forecasting",
    root_dir = "era5/5.625",

    in_vars = ["2m_temperature", "total_precipitation"],
    out_vars = ["2m_temperature"],

    train_start_year = Year(2000),
    val_start_year = Year(2015),
    test_start_year = Year(2017),
    end_year = Year(2018),


    pred_range = Days(3),
    subsample = Hours(6),
    specify_range = True,
    
    min_lat = 26,
    max_lat = 50,
    min_lon = 230,
    max_lon = 310,
    val_lat_start = 35,
    val_lon_start = 0,
    test_lat_start = 35,
    test_lon_start = 0,
    
    batch_size = 128,
    num_workers = 1
)
