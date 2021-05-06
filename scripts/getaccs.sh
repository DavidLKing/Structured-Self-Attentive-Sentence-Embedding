for i in *.log; 
do grep "eval.*valid.*Acc" $i | cut -d ' ' -f 16 > $i.devaccs;
done

ls *.devaccs | perl -pe 's/\n/\t/g' | perl -pe 's/\.log\.devaccs//g' > accs.tsv

# huh, line didn't terminate
echo "" >> accs.tsv

paste *.devaccs >> accs.tsv

rm *.devaccs
