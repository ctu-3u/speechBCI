baseDir = 'F:\\RiceStudy\\speechBCI\\data'

import os

from getSpeechSessionBlocks import getSpeechSessionBlocks
blockLists = getSpeechSessionBlocks()

for sessIdx in range(len(blockLists)):
    sessionName = blockLists[sessIdx][0]
    dataPath = baseDir + '\\competitionData'
    tfRecordFolder = baseDir + '\\derived\\tfRecords\\'+sessionName

