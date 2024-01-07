import React from 'react';

function Navbar() {
  const navbarStyle = {
    backgroundColor: '#2c3e50',
    color: 'white',
    padding: '15px',
    fontWeight: 'bold',
    display: 'flex',
    justifyContent: 'flex-start',
    alignItems: 'center',
  };

  const logoStyle = {
    
    marginRight: '10px',
    height: '20px',
    fit: 'stretch',
  };



  return (
    <div style={navbarStyle}>
      <img src="sudoku.png" alt="Logo" style={logoStyle} />
      <span>SUDOKU SOLVER</span>
    </div>
  );
}

export default Navbar;