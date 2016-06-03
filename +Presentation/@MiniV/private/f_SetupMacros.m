%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Filename: f_SetupMacros.m
%
% Creates macros for the different grasp patterns that control 
% the Joints that make up the movement
%
% Author: Erik Scheme
% Date Created: May 10, 2006
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
function Macros = f_SetupMacros(handles)
 
Macros(1).Description = 'Key Grip';
Macros(1).curPosition = 1;
Macros(1).numPositions = 17;
 
Macros(1).Increment(1).iLink  = [6   9  12  15];
Macros(1).Increment(1).iXYZ  = [3  3  3  3];
Macros(1).Increment(1).iDir = [-1  5  5  5];
Macros(1).Increment(2).iLink  = [6   9  12  15];
Macros(1).Increment(2).iXYZ  = [3  3  3  3];
Macros(1).Increment(2).iDir = [-1  5  5  5];
Macros(1).Increment(3).iLink  = [6   9  12  15  18];
Macros(1).Increment(3).iXYZ  = [3  3  3  3  3];
Macros(1).Increment(3).iDir = [-1  5  5  5  5];
Macros(1).Increment(4).iLink  = [6   9  12  15  18  19  20];
Macros(1).Increment(4).iXYZ  = [3  3  3  3  3  3  3];
Macros(1).Increment(4).iDir = [-1  5  5  5  5  5  5];
Macros(1).Increment(5).iLink  = [6   9  10  11  12  15  16  17  18  19  20];
Macros(1).Increment(5).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(5).iDir = [-1  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(6).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(6).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(6).iDir = [-1  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(7).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(7).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(7).iDir = [-1  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(8).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(8).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(8).iDir = [-1  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(9).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(9).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(9).iDir = [-1  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(10).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(10).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(10).iDir = [-1  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(11).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(11).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(11).iDir = [-1  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(12).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(12).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(12).iDir = [-1  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(13).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(13).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(13).iDir = [-1  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(14).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(14).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(14).iDir = [-2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(15).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(15).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(15).iDir = [-3  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(1).Increment(16).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(1).Increment(16).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(1).Increment(16).iDir = [-2  5  5  5  5  5  5  5  5  5  5  5  5];
 
Macros(2).Description = '3-Jaw Chuck Grip';
Macros(2).curPosition = 1;
Macros(2).numPositions = 12;
 
Macros(2).Increment(1).iLink  = [6   6   7   9  10  11  12  13  14  15  16  17  19  20];
Macros(2).Increment(1).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(2).Increment(1).iDir = [5 -5  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(2).Increment(2).iLink  = [6   6   9  10  11  12  13  14  15  16  17  19  20];
Macros(2).Increment(2).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(2).Increment(2).iDir = [5 -5  5  5  5  5  5  5  5  5  5  5  5];
Macros(2).Increment(3).iLink  = [6   6   9  10  11  12  13  14  15  16  17];
Macros(2).Increment(3).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(2).Increment(3).iDir = [5 -5  5  5  5  5  5  5  5  5  5];
Macros(2).Increment(4).iLink  = [6   6   9  10  11  12  13  14  15  16  17];
Macros(2).Increment(4).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(2).Increment(4).iDir = [5 -5  5  5  5  5  5  5  5  5  5];
Macros(2).Increment(5).iLink  = [6   6   9  10  11  12  13  14  15  16  17];
Macros(2).Increment(5).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(2).Increment(5).iDir = [5 -5  5  5  5  5  5  5  5  5  5];
Macros(2).Increment(6).iLink  = [6   9  10  11  12  13  14  15];
Macros(2).Increment(6).iXYZ  = [3  3  3  3  3  3  3  3];
Macros(2).Increment(6).iDir = [-5  5  5  5  5  5  5  5];
Macros(2).Increment(7).iLink  = [6   9  10  11  12  13  14  15];
Macros(2).Increment(7).iXYZ  = [3  3  3  3  3  3  3  3];
Macros(2).Increment(7).iDir = [-5  5  5  5  5  5  5  5];
Macros(2).Increment(8).iLink  = [9  12];
Macros(2).Increment(8).iXYZ  = [3  3];
Macros(2).Increment(8).iDir = [5  5];
Macros(2).Increment(9).iLink  = [9  12];
Macros(2).Increment(9).iXYZ  = [3  3];
Macros(2).Increment(9).iDir = [5  5];
Macros(2).Increment(10).iLink  = [12];
Macros(2).Increment(10).iXYZ  = [3];
Macros(2).Increment(10).iDir = [5];
Macros(2).Increment(11).iLink  = [12];
Macros(2).Increment(11).iXYZ  = [3];
Macros(2).Increment(11).iDir = [5];
 
Macros(3).Description = 'Power Grip';
Macros(3).curPosition = 1;
Macros(3).numPositions = 14;
 
Macros(3).Increment(1).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(1).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(1).iDir = [1  5  5  5  5  6  6  5  6  6  5  5  5];
Macros(3).Increment(2).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(2).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(2).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(3).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(3).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(3).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(4).iLink  = [6   6   7   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(4).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(4).iDir = [2 -2  2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(5).iLink  = [6   6   7   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(5).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(5).iDir = [2 -2  2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(6).iLink  = [6   6   7   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(6).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(6).iDir = [2 -2  2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(7).iLink  = [6   6   7   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(7).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(7).iDir = [2 -2  2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(8).iLink  = [6   6   7   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(8).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(8).iDir = [2 -2  2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(9).iLink  = [6   6   7   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(9).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(9).iDir = [2 -2  2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(10).iLink  = [6   6   7   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(10).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(10).iDir = [2 -2  2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(11).iLink  = [6   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(11).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(11).iDir = [2 -2  2  1  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(12).iLink  = [6   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(12).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(12).iDir = [2 -2  2  2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(3).Increment(13).iLink  = [6   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(3).Increment(13).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(3).Increment(13).iDir = [2 -2  2  2  5  5  5  5  4  4  5  4  4  5  5  5];
 
 
Macros(4).Description = 'Pinch Grip';
Macros(4).curPosition = 1;
Macros(4).numPositions = 20;
 
Macros(4).Increment(1).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(4).Increment(1).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(4).Increment(1).iDir = [-5  10  10  10  10  10  10  10   9   9];
Macros(4).Increment(2).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(4).Increment(2).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(4).Increment(2).iDir = [-5  10  10  10  10  10  10  10  10  10];
Macros(4).Increment(3).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(4).Increment(3).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(4).Increment(3).iDir = [-5  10  10  10  10  10  10  10  10  10];
Macros(4).Increment(4).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(4).Increment(4).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(4).Increment(4).iDir = [-5  10  10  10  10  10  10  10  10  10];
Macros(4).Increment(5).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(4).Increment(5).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(4).Increment(5).iDir = [-5  10  10  10  10  10  10  10  10  10];
Macros(4).Increment(6).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(4).Increment(6).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(4).Increment(6).iDir = [-5  5  5  5  5  5  5  5  5  5];
Macros(4).Increment(7).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(4).Increment(7).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(4).Increment(7).iDir = [-5  5  5  5  5  5  5  5  5  5];
Macros(4).Increment(8).iLink  = [12  13  14  15  16  17  18  19  20];
Macros(4).Increment(8).iXYZ  = [3  3  3  3  3  3  3  3  3];
Macros(4).Increment(8).iDir = [5  5  5  5  5  5  5  5  5];
Macros(4).Increment(9).iLink  = [12  13  14  15  16  17  18  19  20];
Macros(4).Increment(9).iXYZ  = [3  3  3  3  3  3  3  3  3];
Macros(4).Increment(9).iDir = [5  5  5  5  5  5  5  5  5];
Macros(4).Increment(10).iLink  = [12  15  18];
Macros(4).Increment(10).iXYZ  = [3  3  3];
Macros(4).Increment(10).iDir = [5  5  5];
Macros(4).Increment(11).iLink  = [6   9  12  15  18];
Macros(4).Increment(11).iXYZ  = [2  3  3  3  3];
Macros(4).Increment(11).iDir = [1  5  5  5  5];
Macros(4).Increment(12).iLink  = [6   9  10  11];
Macros(4).Increment(12).iXYZ  = [2  3  3  3];
Macros(4).Increment(12).iDir = [3  5  5  5];
Macros(4).Increment(13).iLink  = [6   9  10  11];
Macros(4).Increment(13).iXYZ  = [2  3  3  3];
Macros(4).Increment(13).iDir = [3  5  5  5];
Macros(4).Increment(14).iLink  = [6   9  10  11];
Macros(4).Increment(14).iXYZ  = [2  3  3  3];
Macros(4).Increment(14).iDir = [3  5  5  5];
Macros(4).Increment(15).iLink  = [6   9  10  11];
Macros(4).Increment(15).iXYZ  = [2  3  3  3];
Macros(4).Increment(15).iDir = [3  5  5  5];
Macros(4).Increment(16).iLink  = [6   9  10  11];
Macros(4).Increment(16).iXYZ  = [2  3  3  3];
Macros(4).Increment(16).iDir = [3  5  5  5];
Macros(4).Increment(17).iLink  = [6   9  10  11];
Macros(4).Increment(17).iXYZ  = [2  3  3  3];
Macros(4).Increment(17).iDir = [3  5  5  5];
Macros(4).Increment(18).iLink  = [6   9  10  11];
Macros(4).Increment(18).iXYZ  = [2  3  3  3];
Macros(4).Increment(18).iDir = [3  5  5  5];
Macros(4).Increment(19).iLink  = [6   9  10  11];
Macros(4).Increment(19).iXYZ  = [2  3  3  3];
Macros(4).Increment(19).iDir = [3  5  5  5];
 
Macros(5).Description = 'Tool Grip';
Macros(5).curPosition = 1;
Macros(5).numPositions = 23;
 
Macros(5).Increment(1).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(1).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(1).iDir = [2 -4  4  5  5  5  5  5  4  5  5];
Macros(5).Increment(2).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(2).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(2).iDir = [1 -4  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(3).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(3).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(3).iDir = [1 -4  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(4).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(4).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(4).iDir = [1 -3  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(5).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(5).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(5).iDir = [1 -3  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(6).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(6).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(6).iDir = [1 -3  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(7).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(7).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(7).iDir = [1 -3  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(8).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(8).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(8).iDir = [1 -3  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(9).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(9).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(9).iDir = [1 -3  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(10).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(10).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(10).iDir = [1 -3  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(11).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(11).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(11).iDir = [1 -3  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(12).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(5).Increment(12).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(5).Increment(12).iDir = [1 -3  5  5  5  5  5  5  5  5  5];
Macros(5).Increment(13).iLink  = [6   6  12  15  18];
Macros(5).Increment(13).iXYZ  = [2  3  3  3  3];
Macros(5).Increment(13).iDir = [1 -3  5  5  5];
Macros(5).Increment(14).iLink  = [6   6  10  11  12  15  18];
Macros(5).Increment(14).iXYZ  = [2  3  3  3  3  3  3];
Macros(5).Increment(14).iDir = [1  -3  10  10   5   5   5];
Macros(5).Increment(15).iLink  = [10  11];
Macros(5).Increment(15).iXYZ  = [3  3];
Macros(5).Increment(15).iDir = [10  10];
Macros(5).Increment(16).iLink  = [10  11];
Macros(5).Increment(16).iXYZ  = [3  3];
Macros(5).Increment(16).iDir = [10  10];
Macros(5).Increment(17).iLink  = [10  11];
Macros(5).Increment(17).iXYZ  = [3  3];
Macros(5).Increment(17).iDir = [10  10];
Macros(5).Increment(18).iLink  = [10  11];
Macros(5).Increment(18).iXYZ  = [3  3];
Macros(5).Increment(18).iDir = [10  10];
Macros(5).Increment(19).iLink  = [10  11];
Macros(5).Increment(19).iXYZ  = [3  3];
Macros(5).Increment(19).iDir = [10  10];
Macros(5).Increment(20).iLink  = [10  11];
Macros(5).Increment(20).iXYZ  = [3  3];
Macros(5).Increment(20).iDir = [10  10];
Macros(5).Increment(21).iLink  = [10  11];
Macros(5).Increment(21).iXYZ  = [3  3];
Macros(5).Increment(21).iDir = [10  10];
Macros(5).Increment(22).iLink  = [10  11];
Macros(5).Increment(22).iXYZ  = [3  3];
Macros(5).Increment(22).iDir = [10  10];
 
 
Macros(6).Description = 'Pointing Grip';
Macros(6).curPosition = 1;
Macros(6).numPositions = 17;
 
Macros(6).Increment(1).iLink  = [6   6  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(1).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(1).iDir = [-1 -4  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(2).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(2).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(2).iDir = [-4  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(3).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(3).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(3).iDir = [-4  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(4).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(4).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(4).iDir = [-4  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(5).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(5).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(5).iDir = [-2  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(6).iLink  = [6];
Macros(6).Increment(6).iXYZ  = [2];
Macros(6).Increment(6).iDir = [1];
Macros(6).Increment(7).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(7).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(7).iDir = [-2  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(8).iLink  = [6  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(8).iXYZ  = [3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(8).iDir = [-2  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(9).iLink  = [6   6   7  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(9).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(9).iDir = [1 -2  1  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(10).iLink  = [6   6   7  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(10).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(10).iDir = [1 -2  1  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(11).iLink  = [6   6   7  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(11).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(11).iDir = [1 -2  1  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(12).iLink  = [6   6   7  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(12).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(12).iDir = [1 -2  1  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(13).iLink  = [6   6   7  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(13).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(13).iDir = [1 -2  1  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(14).iLink  = [6   6   7  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(14).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(14).iDir = [1 -2  1  5  5  5  5  5  5  5  5  5];
Macros(6).Increment(15).iLink  = [6   6   7  12  13  14  15  16  17  18  19  20];
Macros(6).Increment(15).iXYZ  = [2  3  3  3  3  3  3  3  3  3  3  3];
Macros(6).Increment(15).iDir = [3  -4   3  10   5   5  10  10  10  10  10  10];
Macros(6).Increment(16).iLink  = [6   6   7  16  17  19  20];
Macros(6).Increment(16).iXYZ  = [2  3  3  3  3  3  3];
Macros(6).Increment(16).iDir = [1 -2  1  4  4  5  5];
 
 
Macros(7).Description = 'Hook Grip';
Macros(7).curPosition = 1;
Macros(7).numPositions = 15;
 
Macros(7).Increment(1).iLink  = [9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(1).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(1).iDir = [5  5  5  4  4  4  4  4  4  5  5  5];
Macros(7).Increment(2).iLink  = [9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(2).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(2).iDir = [5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(3).iLink  = [9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(3).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(3).iDir = [5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(4).iLink  = [9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(4).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(4).iDir = [5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(5).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(5).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(5).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(6).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(6).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(6).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(7).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(7).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(7).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(8).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(8).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(8).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(9).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(9).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(9).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(10).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(10).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(10).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(11).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(11).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(11).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(12).iLink  = [6   9  10  11  12  13  14  15  16  17  18  19  20];
Macros(7).Increment(12).iXYZ  = [3  3  3  3  3  3  3  3  3  3  3  3  3];
Macros(7).Increment(12).iDir = [2  5  5  5  5  5  5  5  5  5  5  5  5];
Macros(7).Increment(13).iLink  = [6   9  12  15  18];
Macros(7).Increment(13).iXYZ  = [3  3  3  3  3];
Macros(7).Increment(13).iDir = [2  5  5  5  5];
Macros(7).Increment(14).iLink  = [6   9  12  15  18];
Macros(7).Increment(14).iXYZ  = [3  3  3  3  3];
Macros(7).Increment(14).iDir = [2  5  5  5  5];