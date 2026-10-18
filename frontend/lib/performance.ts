// Verified results shown by the interface. Source: step 5b `final` report on the TUSZ v2.0.3 evaluation partition
// (43 patients, 125.2 h, 457 annotated seizures); decision rule frozen on the validation patients BEFORE this evaluation.
// Intervals are 95% patient-level bootstrap intervals. Update these numbers from scores/rapport_5b_final.txt, never by hand elsewhere.
export const PERFORMANCE = {
  corpus: 'TUSZ v2.0.3, partition d’évaluation (43 patients, 125,2 h)',
  sensitivity: '0,66 [0,44 ; 0,81]',
  falseAlarmsPerHour: '1,27 [0,51 ; 2,17]',
  precision: '0,75 [0,61 ; 0,86]',
  lateralizationBalancedAccuracy: '0,44 [0,23 ; 0,55]',
  lateralizationChance: '0,33',
  lateralizationAccuracy: '0,40',
  majorityRate: '0,54',
  oracleBalancedAccuracy: '0,68 [0,51 ; 0,83]',
}

export const SHORT_DISCLAIMER =
  'Prototype de recherche, pas un dispositif médical. Ne pas utiliser pour le diagnostic ni pour un traitement. ' +
  'Le côté prédit l’est par rapport à des labels dérivés d’une règle algorithmique sur les annotations par canal de TUSZ, non relus par un clinicien.'
