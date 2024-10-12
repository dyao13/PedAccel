-- ------------------------------------------------------------- --
--
-- This is a script to add the PIC indexes for PostgreSQL.
--
-- These index definitions should be taken as mere suggestions.
--
-- ------------------------------------------------------------- --

-- -----------
-- ADMISSIONS
-- -----------

CREATE INDEX ADMISSIONS_idx01 ON ADMISSIONS (SUBJECT_ID,HADM_ID);
CREATE INDEX ADMISSIONS_idx02 ON ADMISSIONS (ADMITTIME, DISCHTIME, DEATHTIME);

-- -------------
-- CHARTEVENTS
-- -------------

CREATE INDEX CHARTEVENTS_idx01 ON CHARTEVENTS (SUBJECT_ID, HADM_ID, ICUSTAY_ID);
CREATE INDEX CHARTEVENTS_idx02 ON CHARTEVENTS (ITEMID);
CREATE INDEX CHARTEVENTS_idx03 ON CHARTEVENTS (CHARTTIME, STORETIME);


-- ------------------
-- D_ICD_DIAGNOSES
-- ------------------

CREATE INDEX D_ICD_DIAG_idx01 ON D_ICD_DIAGNOSES (ICD10_CODE_CN);
CREATE INDEX D_ICD_DIAG_idx02 ON D_ICD_DIAGNOSES (ICD10_CODE);


-- ---------
-- D_ITEMS
-- ---------

CREATE INDEX D_ITEMS_idx02 ON D_ITEMS (LABEL);
CREATE INDEX D_ITEMS_idx03 ON D_ITEMS (CATEGORY);

-- -------------
-- D_LABITEMS
-- -------------

CREATE INDEX D_LABITEMS_idx02 ON D_LABITEMS (LABEL, FLUID, CATEGORY);
CREATE INDEX D_LABITEMS_idx03 ON D_LABITEMS (LOINC_CODE);

-- ----------------
-- DIAGNOSES_ICD
-- ----------------

CREATE INDEX DIAGNOSES_ICD_idx01 ON DIAGNOSES_ICD (SUBJECT_ID, HADM_ID);
CREATE INDEX DIAGNOSES_ICD_idx02 ON DIAGNOSES_ICD (ICD10_CODE_CN, SEQ_NUM);

-- ----------------
-- EMR_SYMPTOMS
-- ----------------

CREATE INDEX EMR_SYMPTOMS_idx01 ON EMR_SYMPTOMS (SUBJECT_ID, HADM_ID);
CREATE INDEX EMR_SYMPTOMS_idx02 ON EMR_SYMPTOMS (RECORDTIME);


-- ----------------
-- ICUSTAYS
-- ----------------

CREATE INDEX ICUSTAYS_idx01 ON ICUSTAYS (SUBJECT_ID, HADM_ID);
CREATE INDEX ICUSTAYS_idx02 ON ICUSTAYS (ICUSTAY_ID);
CREATE INDEX ICUSTAYS_idx03 ON ICUSTAYS (LOS);
CREATE INDEX ICUSTAYS_idx04 ON ICUSTAYS (FIRST_CAREUNIT);
CREATE INDEX ICUSTAYS_idx05 ON ICUSTAYS (LAST_CAREUNIT);


-- ------------
-- LABEVENTS
-- ------------

CREATE INDEX LABEVENTS_idx01 ON LABEVENTS (SUBJECT_ID, HADM_ID);
CREATE INDEX LABEVENTS_idx02 ON LABEVENTS (ITEMID);
CREATE INDEX LABEVENTS_idx03 ON LABEVENTS (CHARTTIME);
CREATE INDEX LABEVENTS_idx04 ON LABEVENTS (VALUE, VALUENUM);

-- --------------------
-- MICROBIOLOGYEVENTS
-- --------------------

CREATE INDEX MICROBIOLOGYEVENTS_idx01 ON MICROBIOLOGYEVENTS (SUBJECT_ID, HADM_ID);
CREATE INDEX MICROBIOLOGYEVENTS_idx02 ON MICROBIOLOGYEVENTS (CHARTTIME);
CREATE INDEX MICROBIOLOGYEVENTS_idx03 ON MICROBIOLOGYEVENTS (ORG_ITEMID, AB_ITEMID);

-- -------------
-- OR_EXAM_REPORTS
-- -------------

CREATE INDEX OR_EXAM_REPORTS_idx01 ON OR_EXAM_REPORTS (SUBJECT_ID, HADM_ID);
CREATE INDEX OR_EXAM_REPORTS_idx02 ON OR_EXAM_REPORTS (EXAMTIME);

-- --------------
-- OUTPUTEVENTS
-- --------------

CREATE INDEX OUTPUTEVENTS_idx01 ON OUTPUTEVENTS (SUBJECT_ID, HADM_ID);
CREATE INDEX OUTPUTEVENTS_idx02 ON OUTPUTEVENTS (ICUSTAY_ID);
CREATE INDEX OUTPUTEVENTS_idx03 ON OUTPUTEVENTS (CHARTTIME, STORETIME);
CREATE INDEX OUTPUTEVENTS_idx04 ON OUTPUTEVENTS (ITEMID);
CREATE INDEX OUTPUTEVENTS_idx05 ON OUTPUTEVENTS (VALUE);

-- -----------
-- PATIENTS
-- -----------

CREATE INDEX PATIENTS_idx01 ON PATIENTS (EXPIRE_FLAG);

-- ----------------
-- PRESCRIPTIONS
-- ----------------

CREATE INDEX PRESCRIPTIONS_idx01 ON PRESCRIPTIONS (SUBJECT_ID, HADM_ID);
CREATE INDEX PRESCRIPTIONS_idx02 ON PRESCRIPTIONS (ICUSTAY_ID);
CREATE INDEX PRESCRIPTIONS_idx04 ON PRESCRIPTIONS (DRUG_NAME);
CREATE INDEX PRESCRIPTIONS_idx05 ON PRESCRIPTIONS (STARTDATE, ENDDATE);

-- -----------
-- SURGERY_VITAL_SIGNS
-- -----------

CREATE INDEX SURGERY_VITAL_SIGNS_idx01 ON SURGERY_VITAL_SIGNS (SUBJECT_ID, HADM_ID);
CREATE INDEX SURGERY_VITAL_SIGNS_idx02 ON SURGERY_VITAL_SIGNS (ITEM_NO);
CREATE INDEX SURGERY_VITAL_SIGNS_idx03 ON SURGERY_VITAL_SIGNS (ITEMID);
CREATE INDEX SURGERY_VITAL_SIGNS_idx04 ON SURGERY_VITAL_SIGNS (VALUE);

-- --------------
-- INPUTEVENTS
-- --------------

CREATE INDEX INPUTEVENTS_idx01 ON INPUTEVENTS (SUBJECT_ID, HADM_ID);
CREATE INDEX INPUTEVENTS_idx02 ON INPUTEVENTS (ICUSTAY_ID);
CREATE INDEX INPUTEVENTS_idx03 ON INPUTEVENTS (CHARTTIME, STORETIME);
CREATE INDEX INPUTEVENTS_idx04 ON INPUTEVENTS (AMOUNT);
