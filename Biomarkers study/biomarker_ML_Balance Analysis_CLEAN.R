##Sensitivity analysis on RF for survived/non-survivors to account for dataset imbalance.
library(tidymodels)
library(caret)
library(ranger)
library(e1071)
library(vip)


rm(list=ls()[!(grepl("dat_tidy", ls()))])

dat_3h<-dat_tidy %>% 
  filter(died != "healthy") %>% 
  mutate(died = factor(case_when(died == "Died" ~ 1,
                                 TRUE ~ 0)),
         Sex = case_when(Sex == "Female" ~ 0,
                         TRUE ~ 1)) %>% 
  select(-c("pat_type", "any_covid", "severity")) %>% 
  na.omit()


dat_b<-dat_3h %>% 
  filter(died == 1)

dat_b2<-dat_3h %>% 
  filter(died == 0) %>% 
  sample_n(21)


dat_bc<-bind_rows(dat_b, dat_b2)

# set original parameters for this analysis
params_list = list(seed = 110,
                   prop = 0.66,
                   IDVars = c("VIALID"),
                   ntrees = 10, 
                   ngrid = 20, 
                   ncv = 5,
                   min_n = FALSE,
                   mtry = FALSE)

# Function for getting training and testing models
get_all_models = function(df, params){
  
  # structuring recipe
  rec = cookbook(df)
  
  #get splits, train/test data
  # d_list output is 1) data splits 2) training data 3) test data
  d_list = splitz(df, params)
  
  #tune random forest hyperparams with grid_search from training data
  m_spec = tune_import_og(params) #specify tuning model 
  t_model = tune_folds(d_list[[2]], m_spec, rec, params) #run grid search
  t_metrics = t_model %>% collect_metrics() #pull metrics from grid search
  t_metrics$trees = params$ntrees #pull ntrees from params
  
  #select model with highest roc_auc
  f_params = select_best(t_model, metric = "roc_auc")
  f_model = finalize_mod(f_params, params)
  
  # obtain the model output for the test set
  v_model = valid_mod(d_list[[1]], rec, f_model)
  v_metrics = v_model %>% collect_metrics() %>% 
    cbind(f_params)
  v_metrics$trees = params$ntrees
  
  # capture all model output in a list and return
  list('train_metrics' = t_metrics,
       'train_notes' = t_model$.notes,
       'best_model' = f_model, 
       'test_metrics' = v_metrics)
}

#Prep data function
juicer = function(df){
  cookbook(df) %>%
    juice()
}

####
# make recipe
####
#make recipe w/ model and dataset; specify that ids aren't predictors
#rec is recipe name for other functs. (syntax rec=cookbook(args))
#prep() to update recipe with parameters that don't need data for training
cookbook = function(df){
  recipe(formula = died ~ .,
         data = df) %>%
    update_role(c(VIALID),
                new_role = "ID") #%>%
  #prep()
}

#Split data function
splitz = function(df, params){
  
  #set seed for reproducibility
  set.seed(params$seed)
  
  #make split object
  data_split = initial_split(df,
                             prop = params$prop,
                             strata = "died")
  #create train and test data
  train = training(data_split)
  test = testing(data_split)
  
  list(data_split, train, test)
}


#function for initial tuning model to plug into get_all_models
tune_import_og = function(params){
  
  #make tuning model
  a_model = rand_forest(
    mtry = tune(),
    trees = !!params$ntrees,
    min_n = tune() 
  ) 
  
  a_model %>%
    set_mode("classification") %>%
    set_engine("ranger", 
               seed = !!params$seed)
}                 

#new function for tuning to plug into other functions to be used after og tuning
#
#different from other tuning function because it collects permutation importance;
#   this isn't collected with initial tuning because it's computationally 
#   expensive and only necessary  for final model
tune_import_new = function(params){
  
  #make tuning model
  a_model = rand_forest(
    mtry = !!params$mtry,
    trees = !!params$ntrees,
    min_n = !!params$min_n 
  ) 
  
  
  a_model %>% 
    set_mode("classification") %>%
    set_engine("ranger", 
               importance = "permutation",
               seed = !!params$seed)
}

####
# tune model hyperparameters plus cross validation
####
tune_folds = function(df, model, rec, params){
  tunewf = workflow() %>%
    add_recipe(rec) %>%
    add_model(model)
  
  set.seed(params$seed)
  folds = vfold_cv(df,
                   params$ncv)
  
  #if ngrid is a number use grid search to find best hyperparams
  #after tuning, change ngrid to FALSE to trigger else and cross validate hyperparams
  
  if(params$ngrid){
    set.seed(params$seed)
    tune_grid(
      tunewf,
      resamples = folds,
      grid = params$ngrid,
      metrics = metric_set(pr_auc, roc_auc)
    )
  } else{
    control = control_resamples(save_pred = TRUE)
    set.seed(params$seed)
    fit_resamples(tunewf,
                  folds,
                  metrics = metric_set(pr_auc, roc_auc),
                  control = control)
  }
  
}

#tunes model with old parameters, then updates with new ones
finalize_mod <- function(new_parms, params){
  tune_spec <- tune_import_new(params)
  
  finalize_model(
    tune_spec,
    new_parms
  )
}

#validates model 
#fits data one more time on training data, then validates on test set
#used in fin_mod_import
valid_mod <- function(data_splits, rec, fin_rf){
  final_wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(fin_rf)
  
  final_wf %>%
    last_fit(data_splits,
             metrics = metric_set(pr_auc, roc_auc))
}


#extracts variable importance after tuning model
#used in fin_mod_import
v_important = function(df, final_mod, rec, params){
  import_mod = final_mod
  
  if(pluck(params, 'ngrid')){
    import_mod %<>% 
      set_engine("ranger", 
                 importance = "permutation",
                 seed = !!params$seed)
  }
  
  import_wf = workflow() %>%
    add_recipe(rec) %>% 
    add_model(import_mod)
  
  imp_fit = import_wf %>% 
    fit(df)
  
  pluck(imp_fit, 'fit', 'fit', 'fit', 'variable.importance') %>% 
    sort(decreasing = TRUE)
}


#trains and validates  final model and extracts variable importance
#output is a list:
#  list('train_model' = t_model,
#   'train_metrics' = t_metrics, 
#   'train_predictions' = train_pred_df,
#   'train_variable_importances' = train_var_imps,
#   'test_model' = val_m,
#   'test_metrics' = val_metrics,
#   'test_predictions' = test_pred_df,
#   'test_variable_importances' = test_var_imps)

finmod_import = function(df, params){
  # structuring recipe
  rec = cookbook(df)
  
  # getting splits and dataframes for train/test data
  # d_list output is 1) data splits 2) training data 3) test data
  d_list = splitz(df,params)
  
  # set a RF model specification with designated variables from grid search
  m_spec = tune_import_new(params)
  
  # train model given a recipe and model parameters
  print("Training model...")
  t_model = tune_folds(d_list[[2]], m_spec, rec, params)
  t_metrics = t_model %>% 
    collect_metrics()
  t_metrics$trees = params$ntrees
  
  #map train predictions to patient id
  print("Mapping training predictions...")
  train_pred_df = map(t_model$.predictions,
                      function(x){
                        xRows = x$.row
                        vialid = d_list[[2]] %>% 
                          slice(xRows) %>% 
                          pull(VIALID)
                        cbind(x, "VIALID" = vialid)
                      }) %>% 
    rbindlist(idcol = "cv_split")
  
  train_var_imps = v_important(d_list[[2]], m_spec, rec, params)
  test_var_imps = v_important(d_list[[3]], m_spec, rec, params)
  
  # run the model on the test set
  rf_params = params[c('mtry', 'min_n', 'ntrees')] %>% data.frame()
  
  print("Validating model...")
  val_m = valid_mod(d_list[[1]], rec, m_spec)
  val_metrics = cbind(rf_params, val_m %>% collect_metrics())
  
  #map patient ids to test predictions
  print("Mapping test predictions...")
  test_pred_df = map_dfr(val_m$.predictions, 
                         function(x){
                           vialid = d_list[[3]] %>%  
                             pull(VIALID)
                           cbind(x, "VIALID" = vialid)
                         }) %>% 
    as_tibble()
  
  
  # capture all model output in a list and return
  list('train_model' = t_model,
       'train_metrics' = t_metrics, 
       'train_predictions' = train_pred_df,
       'train_variable_importances' = train_var_imps,
       'test_model' = val_m,
       'test_metrics' = val_metrics,
       'test_predictions' = test_pred_df,
       'test_variable_importances' = test_var_imps)
  
}


#get best parameters from the sample splits
#not used if testing whole dataset -- only for multiple samples/subsets of data
obtain_parameters <- function(L_out){
  controlEvals = map_dfr(1:length(L_out), 
                         function(x) pluck(L_out, x, 'test_metrics')) %>% 
    mutate(uniq_param1 = paste0(mtry, '-', min_n),
           uniq_param2 = paste0(mtry, '-', min_n, '-', .metric))
  
  countEvals = controlEvals %>% filter(.metric == 'roc_auc') %>% 
    count(mtry, min_n, uniq_param1)
  
  maxVal =  countEvals %>% pull(n) %>% max()
  
  rowEvals = countEvals %>% filter(n == maxVal)
  
  if(nrow(rowEvals) > 1){
    rowEvals = controlEvals %>% 
      filter(uniq_param1 %in% rowEvals$uniq_param1) %>% 
      filter(.metric == 'roc_auc') %>% 
      filter(max(.estimate) == .estimate)
    if(nrow(rowEvals) > 1){
      rowEvals = rowEvals %>% slice(1)
    }
  }
  
  rowEvals %>% select(mtry, min_n)
}



####
# Bind predictions to original data
####

bind_preds = function(params, df, fin_rfmodel){
  # getting splits and dataframes for train/test data
  # d_list output is 1) data splits 2) training data 3) test data
  d_list = splitz(df, params)
  
  #add to training data
  p_traindf = inner_join(as.data.frame(d_list[2]),
                         as.data.frame(fin_rfmodel$train_predictions),
                         by = "VIALID") %>%
    rename(type_diag = died.x,
           type_pred = died.y) %>%
    select(-c(cv_split,
              .pred_0,
              .pred_1,
              .row,
              .config))
  
  #add to testing
  p_testdf = inner_join(as.data.frame(d_list[3]),
                        as.data.frame(fin_rfmodel$test_predictions),
                        by = "VIALID") %>%
    rename(type_diag = died.x,
           type_pred = died.y) %>%
    select(-c(.row,
              .config,
              .pred_0,
              .pred_1,
              .row,
              .config))
  
  list(p_traindf, p_testdf)
}


#make function with entire process for running everything 10 times
entirety = function(seed_set, df){
  # set original parameters 
  Params_list = list(seed = seed_set,
                     prop = 0.66,
                     IDVars = c("VIALID"),
                     ngrid = 20,
                     ncv = 5,
                     ntrees = 10,
                     min_n = FALSE,
                     mtry = FALSE)
  
  #grid search to get hyperparameters
  Rf_model = get_all_models(df, params_list)
  print("Grid search complete!")
  
  #get model metrics after grid search
  Rf_mets = pluck(Rf_model,"test_metrics")
  
  #update parameters
  N_params_list = Params_list
  N_params_list$min_n = Rf_mets$min_n[1]
  N_params_list$mtry = Rf_mets$mtry[1]
  N_params_list$ngrid = FALSE
  N_params_list$ntrees = 200
  
  #get final model
  Fin_rfmod = finmod_import(df, N_params_list)
  print("Final model created!")
  
  #binds predicted outcomes and returns list of dataframes [1]train [2]test
  Data_preds = bind_preds(Params_list, df, Fin_rfmod)
  
  list("final model" = Fin_rfmod,
       "data with preds" = Data_preds)
}

####Move on to analysis####
##lived vs died
out1 = entirety(seed_set = 110, df = dat_bc)
out2 = entirety(seed_set = 210, df = dat_bc)
out3 = entirety(seed_set = 310, df = dat_bc)
out4 = entirety(seed_set = 410, df = dat_bc)
out5 = entirety(seed_set = 510, df = dat_bc)
out6 = entirety(seed_set = 610, df = dat_bc)
out7 = entirety(seed_set = 710, df = dat_bc)
out8 = entirety(seed_set = 810, df = dat_bc)
out9 = entirety(seed_set = 910, df = dat_bc)
out10 = entirety(seed_set = 1010, df = dat_bc)

#This will get you the pr_auc and roc_auc for the final models

pluck(out1, 1, 6)
pluck(out2, 1, 6)
pluck(out3, 1, 6)
pluck(out4, 1, 6)
pluck(out5, 1, 6)
pluck(out6, 1, 6)
pluck(out7, 1, 6)
pluck(out8, 1, 6)
pluck(out9, 1, 6)
pluck(out10, 1, 6)
