roc(1).id = 0;
roc(1).name = 'rest';
roc(1).waypoint = [0 1];
roc(1).joints = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27];
roc(1).angles = [
    0.0000,0.3491,0.3840,0.3142,0.0000,0.3491,0.3840,0.3142,0.0000,0.3491,0.3840,0.3142,0.0000,0.3491,0.3840,0.3142,0.0000,1.0472,0.2618,-0.3491;...
    0.0000,0.3491,0.3840,0.3142,0.0000,0.3491,0.3840,0.3142,0.0000,0.3491,0.3840,0.3142,0.0000,0.3491,0.3840,0.3142,0.0000,1.0472,0.2618,-0.3491];

%MPL.writeRocTable('C:\usr\MPL\VulcanX\roc_tables_clinical.xml',roc)
MPL.RocTable.writeRocTable('roc_tables_clinical.xml',roc)
